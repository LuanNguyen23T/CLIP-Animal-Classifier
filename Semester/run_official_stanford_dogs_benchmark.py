"""
Comprehensive Training & Evaluation Benchmark on Stanford Dogs Official Split (12,000 Train / 8,580 Test).
Evaluates:
- Zero-Shot: CLIP Raw, CLIP Handcrafted, CuPL, LaBo-style, Gemma-CBM, Gemma-Centroid, Gemma-ZeroShot
- PEFT Supervised: Baseline CoOp, Gemma-CoOp, Gemma-DualStream, Gemma-VisAdapter
Strict Protocol:
- Stratified 90:10 train/validation split (10,800 train / 1,200 val; 10 images per breed for val).
- Checkpoints and hyperparameters selected strictly at peak validation accuracy.
- Final inference evaluated once on all 8,580 official unobserved test images.
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer
from sklearn.metrics import f1_score

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Executing Stanford Dogs Official Benchmark on device: {device}")

FEAT_DIR = "Semester/cached_features/stanford_dogs_official"
ATTR_FILE = "Semester/attributes/stanford_dogs_attributes.json"
RESULTS_DIR = "Semester/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

with open(os.path.join(FEAT_DIR, "classes.json"), encoding="utf-8") as f:
    classes = json.load(f)
with open(ATTR_FILE, encoding="utf-8") as f:
    attributes = json.load(f)

train_clip = torch.load(os.path.join(FEAT_DIR, "train_clip.pt"), map_location=device).float()
test_clip = torch.load(os.path.join(FEAT_DIR, "test_clip.pt"), map_location=device).float()
train_labels = torch.load(os.path.join(FEAT_DIR, "train_labels.pt"), map_location=device)
test_labels = torch.load(os.path.join(FEAT_DIR, "test_labels.pt"), map_location=device)

num_classes = len(classes)
print(f"Loaded features: Train={train_clip.shape}, Test={test_clip.shape}, Classes={num_classes}")
assert len(train_clip) == 12000
assert len(test_clip) == 8580

# Stratified split: 90% train (10,800), 10% val (1,200)
torch.manual_seed(42)
val_indices, train_indices = [], []
for c in range(num_classes):
    c_idx = (train_labels == c).nonzero(as_tuple=True)[0]
    perm = torch.randperm(len(c_idx))
    n_val = 10 # exactly 10 images per class for validation
    val_indices.append(c_idx[perm[:n_val]])
    train_indices.append(c_idx[perm[n_val:]])

val_idx = torch.cat(val_indices)
train_idx = torch.cat(train_indices)

train_x, train_y = train_clip[train_idx], train_labels[train_idx]
val_x, val_y = train_clip[val_idx], train_labels[val_idx]
print(f"Official Split: Train={len(train_x)}, Val={len(val_x)}, Test={len(test_clip)}")

model_id = "openai/clip-vit-base-patch32"
tokenizer = CLIPTokenizer.from_pretrained(model_id)
clip_model = CLIPModel.from_pretrained(model_id).to(device)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

scale = clip_model.logit_scale.exp().detach()

def compute_metrics(logits, targets):
    preds = logits.argmax(dim=-1)
    acc1 = (preds == targets).float().mean().item() * 100.0
    correct_count = (preds == targets).sum().item()
    top5_preds = logits.topk(5, dim=-1)[1]
    acc5 = (top5_preds == targets.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100.0
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    macro_f1 = f1_score(targets_np, preds_np, average="macro") * 100.0
    return {
        "top1_acc": acc1,
        "correct_count": correct_count,
        "total_count": len(targets),
        "top5_acc": acc5,
        "macro_f1": macro_f1,
        "preds": preds
    }

# ==============================================================================
# 1. TEXT REPRESENTATIONS & ZERO-SHOT BASELINES
# ==============================================================================
print("\n" + "="*70)
print("COMPUTING TEXT EMBEDDINGS & ZERO-SHOT BASELINES")
print("="*70)

# (a) Raw Class Prompts: "a photo of a {breed}"
raw_prompts = [f"a photo of a {c.lower()}." for c in classes]
inps_raw = tokenizer(raw_prompts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
with torch.no_grad():
    W_raw = F.normalize(clip_model.get_text_features(**inps_raw), dim=-1)

# (b) Handcrafted Prompt Ensemble
templates_handcrafted = [
    "a photo of a {c}, a type of dog.",
    "a photo of the {c}.",
    "a photo of a clean {c}.",
    "a close-up photo of a {c}.",
    "a cropped photo of the {c}.",
    "a photo of my {c}."
]
w_hand_list = []
for c in classes:
    p_c = [t.format(c=c.lower()) for t in templates_handcrafted]
    inps = tokenizer(p_c, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
    with torch.no_grad():
        tf = F.normalize(clip_model.get_text_features(**inps), dim=-1)
        w_hand_list.append(F.normalize(tf.mean(dim=0, keepdim=True), dim=-1))
W_handcrafted = torch.cat(w_hand_list, dim=0)

# (c) Morphological Concept Bank (600 concepts across 5 axes)
concept_templates = {
    "head_and_ears": "head and ears of {name}: {desc}",
    "eyes": "eyes of {name}: {desc}",
    "coat": "coat texture and markings of {name}: {desc}",
    "body": "body structure of {name}: {desc}",
    "distinguishing_mark": "distinguishing trait of {name}: {desc}"
}

cbm_weights_list = []
cupl_weights_list = []

for c in classes:
    attr = attributes[c]["attributes"]
    prompts_5 = [
        concept_templates["head_and_ears"].format(name=c, desc=attr["head_and_ears"]),
        concept_templates["eyes"].format(name=c, desc=attr["eyes"]),
        concept_templates["coat"].format(name=c, desc=attr["coat"]),
        concept_templates["body"].format(name=c, desc=attr["body"]),
        concept_templates["distinguishing_mark"].format(name=c, desc=attr["distinguishing_mark"])
    ]
    inps = tokenizer(prompts_5, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
    with torch.no_grad():
        f5 = F.normalize(clip_model.get_text_features(**inps), dim=-1)
        cbm_weights_list.append(f5)
        
    cupl_p = [
        f"a photo of a {c.lower()}, which has {attr['head_and_ears']}.",
        f"a photo of a {c.lower()} with {attr['coat']}.",
        f"a photo of a {c.lower()} displaying {attr['distinguishing_mark']}.",
        f"a photo of a {c.lower()}, a dog with {attr['body']}.",
        f"a close-up photo of a {c.lower()} showing its {attr['eyes']}."
    ]
    inps_cupl = tokenizer(cupl_p, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
    with torch.no_grad():
        tf_cupl = F.normalize(clip_model.get_text_features(**inps_cupl), dim=-1)
        cupl_weights_list.append(F.normalize(tf_cupl.mean(dim=0, keepdim=True), dim=-1))

cbm_weights = torch.stack(cbm_weights_list, dim=0) # [120, 5, 512]
W_centroid = F.normalize(cbm_weights.mean(dim=1), dim=-1) # [120, 512]
W_cupl = torch.cat(cupl_weights_list, dim=0) # [120, 512]

# Zero-Shot Evaluations
with torch.no_grad():
    logits_raw = (test_clip @ W_raw.t()) * scale
    res_raw = compute_metrics(logits_raw, test_labels)
    
    logits_hand = (test_clip @ W_handcrafted.t()) * scale
    res_hand = compute_metrics(logits_hand, test_labels)
    
    logits_cupl = (test_clip @ W_cupl.t()) * scale
    res_cupl = compute_metrics(logits_cupl, test_labels)
    
    sims = torch.einsum("nd,ckd->nck", test_clip, cbm_weights) # [8580, 120, 5]
    logits_cbm = sims.mean(dim=-1) * scale
    res_cbm = compute_metrics(logits_cbm, test_labels)
    
    logits_labo = (sims.max(dim=-1)[0] + sims.mean(dim=-1)) * 0.5 * scale
    res_labo = compute_metrics(logits_labo, test_labels)
    
    logits_cent = (test_clip @ W_centroid.t()) * scale
    res_cent = compute_metrics(logits_cent, test_labels)
    
    logits_zs = logits_hand + 0.5 * logits_cent
    res_zs = compute_metrics(logits_zs, test_labels)

print(f"Zero-Shot Results on Official Test (8,580 images):")
print(f"  • CLIP Raw:         {res_raw['top1_acc']:.2f}% ({res_raw['correct_count']}/8580) | Top-5: {res_raw['top5_acc']:.2f}% | F1: {res_raw['macro_f1']:.2f}%")
print(f"  • CLIP Handcrafted: {res_hand['top1_acc']:.2f}% ({res_hand['correct_count']}/8580) | Top-5: {res_hand['top5_acc']:.2f}% | F1: {res_hand['macro_f1']:.2f}%")
print(f"  • CuPL:             {res_cupl['top1_acc']:.2f}% ({res_cupl['correct_count']}/8580) | Top-5: {res_cupl['top5_acc']:.2f}% | F1: {res_cupl['macro_f1']:.2f}%")
print(f"  • LaBo-style:       {res_labo['top1_acc']:.2f}% ({res_labo['correct_count']}/8580) | Top-5: {res_labo['top5_acc']:.2f}% | F1: {res_labo['macro_f1']:.2f}%")
print(f"  • Gemma-CBM:        {res_cbm['top1_acc']:.2f}% ({res_cbm['correct_count']}/8580) | Top-5: {res_cbm['top5_acc']:.2f}% | F1: {res_cbm['macro_f1']:.2f}%")
print(f"  • Gemma-Centroid:   {res_cent['top1_acc']:.2f}% ({res_cent['correct_count']}/8580) | Top-5: {res_cent['top5_acc']:.2f}% | F1: {res_cent['macro_f1']:.2f}%")
print(f"  • Gemma-ZeroShot:   {res_zs['top1_acc']:.2f}% ({res_zs['correct_count']}/8580) | Top-5: {res_zs['top5_acc']:.2f}% | F1: {res_zs['macro_f1']:.2f}%")

# ==============================================================================
# 2. CONTINUOUS PROMPT TUNING (CoOp, Gemma-CoOp, Gemma-DualStream)
# ==============================================================================
print("\n" + "="*70)
print("TRAINING CONTINUOUS PROMPT PEFT MODELS (CoOp, Gemma-CoOp, Gemma-DualStream)")
print("="*70)

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
        padding_len = 77 - len(seq)
        if padding_len > 0:
            seq = torch.cat([seq, torch.zeros(padding_len, dtype=torch.long)])
        coop_input_ids.append(seq)
    return torch.stack(coop_input_ids).to(device)

texts_base = [f"{c.lower()}." for c in classes]
texts_gemma = []
for c in classes:
    attr = attributes[c]["attributes"]
    trait = attr["distinguishing_mark"].split(".")[0].strip()
    texts_gemma.append(f"{c.lower()}, dog recognized by {trait}.")

ids_base = make_coop_ids(texts_base, num_ctx=4)
ids_gemma = make_coop_ids(texts_gemma, num_ctx=4)

class CoOpClassifierModel(nn.Module):
    def __init__(self, clip_base, num_ctx=4):
        super().__init__()
        self.clip = clip_base
        self.num_ctx = num_ctx
        self.ctx = nn.Parameter(torch.randn(num_ctx, 512, device=device) * 0.02)
        
    def get_text_features(self, input_ids):
        def custom_forward(ids, *args, **kwargs):
            embeds = _ORIG_TOKEN_EMBEDDING_FORWARD(ids, *args, **kwargs).clone()
            for i in range(self.num_ctx):
                mask = (ids == (49400 + i))
                if mask.any():
                    embeds[mask] = self.ctx[i].to(embeds.dtype)
            return embeds
            
        self.clip.text_model.embeddings.token_embedding.forward = custom_forward
        try:
            tf = self.clip.get_text_features(input_ids=input_ids)
        finally:
            self.clip.text_model.embeddings.token_embedding.forward = _ORIG_TOKEN_EMBEDDING_FORWARD
        return F.normalize(tf, dim=-1)

def train_coop_model(coop_ids, name="CoOp", epochs=50, lr=0.005, seed=42):
    torch.manual_seed(seed)
    model = CoOpClassifierModel(clip_model, num_ctx=4).to(device)
    optimizer = torch.optim.AdamW([model.ctx], lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_v = 0.0
    best_epoch = 0
    best_w = None
    
    for ep in range(1, epochs + 1):
        model.train()
        tf = model.get_text_features(coop_ids)
        logits = (train_x @ tf.t()) * scale
        loss = criterion(logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        model.eval()
        with torch.no_grad():
            v_logits = (val_x @ tf.t()) * scale
            v_acc = (v_logits.argmax(-1) == val_y).float().mean().item() * 100.0
            if v_acc > best_v:
                best_v = v_acc
                best_epoch = ep
                best_w = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == epochs:
            print(f"[{name}] Epoch {ep:02d}/{epochs:02d} | Val Acc: {v_acc:.2f}% (Best: {best_v:.2f}% at Ep {best_epoch})")
            
    model.load_state_dict(best_w)
    return model, best_v, best_epoch

print("\n--- Training Standard CoOp on Official 10,800 Train Split ---")
model_base, val_acc_base, ep_base = train_coop_model(ids_base, name="Baseline CoOp", epochs=50, lr=0.005)

print("\n--- Training Gemma-CoOp on Official 10,800 Train Split ---")
model_gemma, val_acc_gemma, ep_gemma = train_coop_model(ids_gemma, name="Gemma-CoOp", epochs=50, lr=0.005)

with torch.no_grad():
    tf_base = model_base.get_text_features(ids_base)
    tf_gemma = model_gemma.get_text_features(ids_gemma)
    
    logits_base_test = (test_clip @ tf_base.t()) * scale
    res_coop_test = compute_metrics(logits_base_test, test_labels)
    
    logits_gemma_test = (test_clip @ tf_gemma.t()) * scale
    res_gcoop_test = compute_metrics(logits_gemma_test, test_labels)
    
    # Gemma-DualStream: grid search beta on Validation Set (1,200 images)
    val_coop_logits = (val_x @ tf_gemma.t()) * scale
    val_cent_logits = (val_x @ W_centroid.t()) * scale
    
    best_beta = 0.5
    best_val_dual = 0.0
    for b_cand in [0.1, 0.2, 0.35, 0.5, 0.7, 1.0]:
        v_blend = val_coop_logits + b_cand * val_cent_logits
        v_acc = (v_blend.argmax(-1) == val_y).float().mean().item() * 100.0
        if v_acc > best_val_dual:
            best_val_dual = v_acc
            best_beta = b_cand
            
    print(f"\nOptimal DualStream blending beta on Val: {best_beta} (Val Acc = {best_val_dual:.2f}%)")
    
    logits_cent_test = (test_clip @ W_centroid.t()) * scale
    logits_dual_test = logits_gemma_test + best_beta * logits_cent_test
    res_dual_test = compute_metrics(logits_dual_test, test_labels)

print(f"\nPrompt PEFT Results on Official Test (8,580 images):")
print(f"  • Standard CoOp:     {res_coop_test['top1_acc']:.2f}% ({res_coop_test['correct_count']}/8580) | Top-5: {res_coop_test['top5_acc']:.2f}% | F1: {res_coop_test['macro_f1']:.2f}%")
print(f"  • Gemma-CoOp:        {res_gcoop_test['top1_acc']:.2f}% ({res_gcoop_test['correct_count']}/8580) | Top-5: {res_gcoop_test['top5_acc']:.2f}% | F1: {res_gcoop_test['macro_f1']:.2f}%")
print(f"  • Gemma-DualStream:  {res_dual_test['top1_acc']:.2f}% ({res_dual_test['correct_count']}/8580) | Top-5: {res_dual_test['top5_acc']:.2f}% | F1: {res_dual_test['macro_f1']:.2f}%")

# ==============================================================================
# 3. RESIDUAL VISUAL ADAPTER (Gemma-VisAdapter, 65,536 parameters)
# ==============================================================================
print("\n" + "="*70)
print("TRAINING RESIDUAL VISUAL ADAPTER (Gemma-VisAdapter, 65,536 parameters)")
print("="*70)

class VisualAdapterMLP(nn.Module):
    def __init__(self, d=512, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden_dim, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, d, bias=False)
        nn.init.zeros_(self.fc2.weight)
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

torch.manual_seed(42)
vis_adapter = VisualAdapterMLP(d=512, hidden_dim=64).to(device)
opt_vis = torch.optim.AdamW(vis_adapter.parameters(), lr=1e-3, weight_decay=1e-4)
alpha = 0.2
criterion = nn.CrossEntropyLoss()

best_v_vis = 0.0
best_ep_vis = 0
best_w_vis = None
epochs_vis = 40

for ep in range(1, epochs_vis + 1):
    vis_adapter.train()
    delta_x = vis_adapter(train_x)
    adapted_x = F.normalize((1.0 - alpha) * train_x + alpha * delta_x, dim=-1)
    logits_train = (adapted_x @ W_centroid.t()) * scale
    loss = criterion(logits_train, train_y)
    
    opt_vis.zero_grad()
    loss.backward()
    opt_vis.step()
    
    vis_adapter.eval()
    with torch.no_grad():
        v_delta = vis_adapter(val_x)
        v_adapt = F.normalize((1.0 - alpha) * val_x + alpha * v_delta, dim=-1)
        v_logits = (v_adapt @ W_centroid.t()) * scale
        v_acc = (v_logits.argmax(-1) == val_y).float().mean().item() * 100.0
        if v_acc > best_v_vis:
            best_v_vis = v_acc
            best_ep_vis = ep
            best_w_vis = {k: v.cpu().clone() for k, v in vis_adapter.state_dict().items()}
            
    if ep % 10 == 0 or ep == epochs_vis:
        print(f"[Gemma-VisAdapter] Epoch {ep:02d}/{epochs_vis:02d} | Val Acc: {v_acc:.2f}% (Best: {best_v_vis:.2f}% at Ep {best_ep_vis})")

vis_adapter.load_state_dict(best_w_vis)
vis_adapter.eval()

with torch.no_grad():
    t_delta = vis_adapter(test_clip)
    t_adapt = F.normalize((1.0 - alpha) * test_clip + alpha * t_delta, dim=-1)
    logits_vis_test = (t_adapt @ W_centroid.t()) * scale
    res_vis_test = compute_metrics(logits_vis_test, test_labels)

print(f"\nGemma-VisAdapter on Official Test (8,580 images):")
print(f"  • Top-1 Acc: {res_vis_test['top1_acc']:.2f}% ({res_vis_test['correct_count']}/8580) | Top-5: {res_vis_test['top5_acc']:.2f}% | F1: {res_vis_test['macro_f1']:.2f}%")

# Save all results
final_results = {
    "benchmark": "Stanford Dogs (Official Split: 12,000 Train / 8,580 Test)",
    "partition_details": {
        "official_train": 12000,
        "official_test": 8580,
        "internal_train_split": len(train_x),
        "internal_val_split": len(val_x),
        "num_classes": num_classes
    },
    "zero_shot_baselines": {
        "clip_raw": res_raw,
        "clip_handcrafted": res_hand,
        "cupl": res_cupl,
        "labo_style": res_labo,
        "gemma_cbm": res_cbm,
        "gemma_centroid": res_cent,
        "gemma_zeroshot": res_zs
    },
    "peft_models": {
        "coop_baseline": {
            "val_acc": val_acc_base,
            "best_epoch": ep_base,
            "trainable_params": 2048,
            "test_results": res_coop_test
        },
        "gemma_coop": {
            "val_acc": val_acc_gemma,
            "best_epoch": ep_gemma,
            "trainable_params": 2048,
            "test_results": res_gcoop_test
        },
        "gemma_dualstream": {
            "val_acc": best_val_dual,
            "best_beta": best_beta,
            "trainable_params": 2048,
            "test_results": res_dual_test
        },
        "gemma_visadapter": {
            "val_acc": best_v_vis,
            "best_epoch": best_ep_vis,
            "alpha": alpha,
            "trainable_params": 65536,
            "test_results": res_vis_test
        }
    }
}

# Clean preds tensor from json
for cat in ["zero_shot_baselines"]:
    for k in final_results[cat]:
        if "preds" in final_results[cat][k]:
            del final_results[cat][k]["preds"]

for k in final_results["peft_models"]:
    if "test_results" in final_results["peft_models"][k]:
        if "preds" in final_results["peft_models"][k]["test_results"]:
            del final_results["peft_models"][k]["test_results"]["preds"]

out_json = os.path.join(RESULTS_DIR, "verified_stanford_dogs_official_split.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=2)

print("\n" + "="*70)
print(f"BENCHMARK COMPLETED AND SAVED TO: {out_json}")
print("="*70)
