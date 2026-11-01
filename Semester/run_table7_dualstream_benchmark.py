"""
Few-Shot Scaling Benchmark for Table 7: Gemma-DualStream vs Baselines.
Evaluates few-shot budgets: K in {1, 2, 4, 8, 16, "full"} across 3 random seeds (1, 2, 3).
Evaluates both single-view (Raw) and 5-View TTA on the official Oxford-IIIT Pet test split (3,669 images).
Results are computed genuinely on hardware and saved directly via json.dump().
"""

import os
import json
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"=== Few-Shot Table 7 Benchmark Running on Device: {device} ===")

SAVE_DIR = "Semester/cached_features/oxford_pets"
ATTR_FILE = "Semester/attributes/oxford_pets_attributes.json"
RESULTS_DIR = "Semester/results"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "verified_table7_dualstream_benchmark.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. LOAD FEATURES & ATTRIBUTES
with open(os.path.join(SAVE_DIR, "classes.json")) as f:
    classes = json.load(f)
with open(ATTR_FILE) as f:
    attributes = json.load(f)

train_clip = torch.load(os.path.join(SAVE_DIR, "train_clip.pt"), map_location=device).float()
test_clip = torch.load(os.path.join(SAVE_DIR, "test_clip.pt"), map_location=device).float()
test_tta = torch.load("dataset/cached/test_features_tta.pt", map_location=device).float()
train_labels = torch.load(os.path.join(SAVE_DIR, "train_labels.pt"), map_location=device)
test_labels = torch.load(os.path.join(SAVE_DIR, "test_labels.pt"), map_location=device)

num_classes = len(classes)
print(f"Loaded Oxford Pets: {num_classes} classes, {len(train_clip)} train, {len(test_labels)} test images.")

# 2. LOAD CLIP MODEL & TOKENIZER
model_id = "openai/clip-vit-base-patch32"
tokenizer = CLIPTokenizer.from_pretrained(model_id)
clip_model = CLIPModel.from_pretrained(model_id).to(device)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False
scale = clip_model.logit_scale.exp().detach()

# 3. BUILD 8-HIERARCHICAL CENTROID BANK (W_centroid)
centroids_ox = []
for c in classes:
    human_name = c
    key = c.lower().replace(" ", "_")
    b_data = attributes[key]
    species = b_data.get("species", "animal")
    attrs = b_data.get("attributes", {})
    prompts_breed = [
        f"a clear photo of a {human_name}, a type of {species}.",
        f"a photo of a {species} belonging to the {human_name} breed.",
        f"a photo of a {human_name} with {attrs.get('head_and_ears', '')}.",
        f"a photo showing the eyes and facial expression of a {human_name}: {attrs.get('eyes', '')}.",
        f"a photo highlighting the coat and fur markings of a {human_name}: {attrs.get('coat', '')}.",
        f"a photo showing the posture and body structure of a {human_name}: {attrs.get('body', '')}.",
        f"a distinguishing photograph of a {human_name}, characterized by {attrs.get('distinguishing_mark', '')}.",
        f"a high quality photo of a {human_name} {species} with {attrs.get('coat', '')} and {attrs.get('head_and_ears', '')}."
    ]
    toks = tokenizer(prompts_breed, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        embeds = F.normalize(clip_model.get_text_features(**toks), dim=-1)
        centroid = F.normalize(embeds.mean(dim=0, keepdim=True), dim=-1)
        centroids_ox.append(centroid)
W_ox_centroid = torch.cat(centroids_ox, dim=0) # [37, 512]

# Zero-shot accuracy for reference
with torch.no_grad():
    zs_raw_preds = (test_clip @ W_ox_centroid.t() * scale).argmax(-1)
    zs_tta_preds = (test_tta @ W_ox_centroid.t() * scale).argmax(-1)
    zs_raw_acc = (zs_raw_preds == test_labels).float().mean().item() * 100.0
    zs_tta_acc = (zs_tta_preds == test_labels).float().mean().item() * 100.0
print(f"Zero-shot Gemma-Centroid: Raw={zs_raw_acc:.2f}%, 5-View TTA={zs_tta_acc:.2f}%")

# 4. PREPARE COOP PROMPT TOKENS
_ORIG_TOKEN_EMBEDDING_FORWARD = clip_model.text_model.embeddings.token_embedding.forward

def make_coop_ids(texts, num_ctx=4):
    coop_input_ids = []
    for txt in texts:
        tokens = tokenizer(txt, return_tensors="pt")["input_ids"][0]
        sos = tokens[:1]
        body = tokens[1:-1]
        eos = tokens[-1:]
        ctx = torch.tensor([49400 + i for i in range(num_ctx)])
        seq = torch.cat([sos, ctx, body, eos])
        if len(seq) > 77:
            seq = torch.cat([sos, ctx, body[:77 - 2 - num_ctx], eos])
        pad_len = 77 - len(seq)
        if pad_len > 0:
            seq = torch.cat([seq, torch.zeros(pad_len, dtype=torch.long)])
        coop_input_ids.append(seq)
    return torch.stack(coop_input_ids).to(device)

texts_coop_base = [f"{c}" for c in classes]
texts_coop_gemma = []
for c in classes:
    k = c.lower().replace(" ", "_")
    d = attributes[k]
    mark = d.get('attributes', {}).get('distinguishing_mark', '')
    texts_coop_gemma.append(f"{c}, {d.get('species', 'pet')}, recognized by {mark}")

ids_coop_base = make_coop_ids(texts_coop_base, num_ctx=4)
ids_coop_gemma = make_coop_ids(texts_coop_gemma, num_ctx=4)

class CoOpClassifierModel(nn.Module):
    def __init__(self, clip_model, num_ctx=4):
        super().__init__()
        self.clip_model = clip_model
        self.num_ctx = num_ctx
        self.ctx = nn.Parameter(torch.randn(num_ctx, 512, device=device) * 0.02)

    def get_text_features(self, input_ids):
        self.clip_model.eval()
        def custom_forward(ids, *args, **kwargs):
            embeds = _ORIG_TOKEN_EMBEDDING_FORWARD(ids, *args, **kwargs).clone()
            for i in range(self.num_ctx):
                mask = (ids == (49400 + i))
                if mask.any():
                    embeds[mask] = self.ctx[i].to(embeds.dtype)
            return embeds
        self.clip_model.text_model.embeddings.token_embedding.forward = custom_forward
        try:
            tf = self.clip_model.get_text_features(input_ids=input_ids)
        finally:
            self.clip_model.text_model.embeddings.token_embedding.forward = _ORIG_TOKEN_EMBEDDING_FORWARD
        return F.normalize(tf, dim=-1)

def sample_stratified(feats, labels, k, num_classes=37, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if k == "full":
        return feats, labels
    selected_indices = []
    for c in range(num_classes):
        c_idx = (labels == c).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(c_idx))[:k]
        selected_indices.append(c_idx[perm])
    idx = torch.cat(selected_indices)
    return feats[idx], labels[idx]

def train_and_eval_coop(bx, by, coop_ids, epochs=30, lr=0.005, seed=42):
    torch.manual_seed(seed)
    model = CoOpClassifierModel(clip_model, num_ctx=4).to(device)
    optimizer = torch.optim.AdamW([model.ctx], lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for ep in range(epochs):
        clip_model.eval()
        tf_train = model.get_text_features(coop_ids)
        logits_tr = (bx @ tf_train.t()) * scale
        loss = criterion(logits_tr, by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        clip_model.eval()
        tf = model.get_text_features(coop_ids)
    return tf

# 5. EXECUTION LOOP ACROSS SHOTS AND SEEDS
shots = [1, 2, 4, 8, 16, "full"]
seeds = [1, 2, 3]

models_evaluated = [
    "CLIP + CoOp",
    "Gemma-CoOp",
    "Gemma-DualStream (Raw)",
    "Gemma-DualStream (5-View TTA)"
]

results = {
    "benchmark": "Table 7 Few-Shot Scaling on Oxford-IIIT Pet",
    "zero_shot_reference": {
        "Gemma-Centroid_Raw": zs_raw_acc,
        "Gemma-Centroid_TTA": zs_tta_acc
    },
    "seeds": seeds,
    "shots": shots,
    "data": {m: {} for m in models_evaluated}
}

print("\n" + "=" * 80)
print("STARTING FEW-SHOT TRAINING & EVALUATION ACROSS 3 SEEDS")
print("=" * 80)

for k in shots:
    shot_key = f"{k}_shot" if k != "full" else "full_trainval"
    print(f"\n>>> Running Budget: {shot_key} <<<")
    
    coop_base_raw_runs = []
    coop_base_tta_runs = []
    gemma_coop_raw_runs = []
    gemma_coop_tta_runs = []
    dualstream_raw_runs = []
    dualstream_tta_runs = []
    
    for s in seeds:
        t0 = time.time()
        bx, by = sample_stratified(train_clip, train_labels, k, num_classes=num_classes, seed=s)
        
        # 1. Baseline CoOp
        tf_base = train_and_eval_coop(bx, by, ids_coop_base, epochs=30, lr=0.005, seed=s)
        with torch.no_grad():
            l_base_s = (test_clip @ tf_base.t()) * scale
            acc_base_s = (l_base_s.argmax(-1) == test_labels).float().mean().item() * 100.0
            l_base_tta = (test_tta @ tf_base.t()) * scale
            acc_base_tta = (l_base_tta.argmax(-1) == test_labels).float().mean().item() * 100.0
        coop_base_raw_runs.append(round(acc_base_s, 2))
        coop_base_tta_runs.append(round(acc_base_tta, 2))
        
        # 2. Gemma-CoOp & Gemma-DualStream
        tf_gemma = train_and_eval_coop(bx, by, ids_coop_gemma, epochs=30, lr=0.005, seed=s)
        with torch.no_grad():
            l_gemma_s = (test_clip @ tf_gemma.t()) * scale
            acc_gemma_s = (l_gemma_s.argmax(-1) == test_labels).float().mean().item() * 100.0
            l_gemma_tta = (test_tta @ tf_gemma.t()) * scale
            acc_gemma_tta = (l_gemma_tta.argmax(-1) == test_labels).float().mean().item() * 100.0
            
            # DualStream = Gemma-CoOp + 0.5 * Gemma-Centroid
            l_dual_s = l_gemma_s + 0.5 * (test_clip @ W_ox_centroid.t()) * scale
            acc_dual_s = (l_dual_s.argmax(-1) == test_labels).float().mean().item() * 100.0
            l_dual_tta = l_gemma_tta + 0.5 * (test_tta @ W_ox_centroid.t()) * scale
            acc_dual_tta = (l_dual_tta.argmax(-1) == test_labels).float().mean().item() * 100.0
            
        gemma_coop_raw_runs.append(round(acc_gemma_s, 2))
        gemma_coop_tta_runs.append(round(acc_gemma_tta, 2))
        dualstream_raw_runs.append(round(acc_dual_s, 2))
        dualstream_tta_runs.append(round(acc_dual_tta, 2))
        
        elapsed = time.time() - t0
        print(f"  [Seed {s}] CoOp TTA={acc_base_tta:.2f}% | Gemma-CoOp TTA={acc_gemma_tta:.2f}% | DualStream Raw={acc_dual_s:.2f}%, TTA={acc_dual_tta:.2f}% ({elapsed:.1f}s)")
        
    def stats(lst):
        return {"runs": lst, "mean": round(float(np.mean(lst)), 2), "std": round(float(np.std(lst)), 2)}
        
    results["data"]["CLIP + CoOp"][shot_key] = stats(coop_base_tta_runs)
    results["data"]["Gemma-CoOp"][shot_key] = stats(gemma_coop_tta_runs)
    results["data"]["Gemma-DualStream (Raw)"][shot_key] = stats(dualstream_raw_runs)
    results["data"]["Gemma-DualStream (5-View TTA)"][shot_key] = stats(dualstream_tta_runs)
    
    print(f"Summary for {shot_key}:")
    print(f"  CLIP + CoOp (TTA):         {results['data']['CLIP + CoOp'][shot_key]['mean']} ± {results['data']['CLIP + CoOp'][shot_key]['std']}%")
    print(f"  Gemma-CoOp (TTA):          {results['data']['Gemma-CoOp'][shot_key]['mean']} ± {results['data']['Gemma-CoOp'][shot_key]['std']}%")
    print(f"  Gemma-DualStream (Raw):    {results['data']['Gemma-DualStream (Raw)'][shot_key]['mean']} ± {results['data']['Gemma-DualStream (Raw)'][shot_key]['std']}%")
    print(f"  Gemma-DualStream (TTA):    {results['data']['Gemma-DualStream (5-View TTA)'][shot_key]['mean']} ± {results['data']['Gemma-DualStream (5-View TTA)'][shot_key]['std']}%")

# 6. SAVE COMPLETE RESULTS TO DISK DIRECTLY
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 80)
print(f"DONE! Verified benchmark successfully saved directly to: {OUTPUT_FILE}")
print("=" * 80)

