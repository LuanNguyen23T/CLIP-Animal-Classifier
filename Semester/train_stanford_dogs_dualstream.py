"""
Training and Evaluation Suite for Gemma-CoOp and Gemma-DualStream on Stanford Dogs.
Strict Protocol:
- 90:10 stratified train/val split (13,024 train, 1,382 val).
- Checkpoint selection exclusively at peak validation accuracy.
- Single evaluation on the 6,174 unobserved test samples.
- Zero test snooping/leakage.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer
from sklearn.metrics import f1_score

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Executing Stanford Dogs DualStream on device: {device}")

SAVE_DIR = "Semester/cached_features/stanford_dogs"
ATTR_FILE = "Semester/attributes/stanford_dogs_attributes.json"
RESULTS_DIR = "Semester/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

with open(os.path.join(SAVE_DIR, "classes.json")) as f:
    classes = json.load(f)
with open(ATTR_FILE, encoding="utf-8") as f:
    attributes = json.load(f)

train_clip = torch.load(os.path.join(SAVE_DIR, "train_clip.pt"), map_location=device).float()
test_clip = torch.load(os.path.join(SAVE_DIR, "test_clip.pt"), map_location=device).float()
train_labels = torch.load(os.path.join(SAVE_DIR, "train_labels.pt"), map_location=device)
test_labels = torch.load(os.path.join(SAVE_DIR, "test_labels.pt"), map_location=device)

num_classes = len(classes)

# Stratified split: 90% train, 10% val
torch.manual_seed(42)
val_indices, train_indices = [], []
for c in range(num_classes):
    c_idx = (train_labels == c).nonzero(as_tuple=True)[0]
    perm = torch.randperm(len(c_idx))
    n_val = max(1, int(len(c_idx) * 0.1))
    val_indices.append(c_idx[perm[:n_val]])
    train_indices.append(c_idx[perm[n_val:]])

val_idx = torch.cat(val_indices)
train_idx = torch.cat(train_indices)

train_x, train_y = train_clip[train_idx], train_labels[train_idx]
val_x, val_y = train_clip[val_idx], train_labels[val_idx]
print(f"Stanford Dogs Split: Train={len(train_x)}, Val={len(val_x)}, Test={len(test_clip)}")

# Load CLIP model and tokenizer
model_id = "openai/clip-vit-base-patch32"
tokenizer = CLIPTokenizer.from_pretrained(model_id)
clip_model = CLIPModel.from_pretrained(model_id).to(device)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

scale = clip_model.logit_scale.exp().detach()

# Helper metric calculation
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

# Build Centroid Bank (600 concepts across 5 axes)
concept_templates = {
    "head_and_ears": "head and ears of {name}: {desc}",
    "eyes": "eyes of {name}: {desc}",
    "coat": "coat texture and markings of {name}: {desc}",
    "body": "body structure of {name}: {desc}",
    "distinguishing_mark": "distinguishing trait of {name}: {desc}"
}

cbm_weights = []
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
        cbm_weights.append(f5)

cbm_weights = torch.stack(cbm_weights, dim=0) # [120, 5, 512]
W_centroid = F.normalize(cbm_weights.mean(dim=1), dim=-1) # [120, 512]

# Prepare CoOp text conditioning tokens
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

_ORIG_TOKEN_EMBEDDING_FORWARD = clip_model.text_model.embeddings.token_embedding.forward

class CoOpClassifierModel(nn.Module):
    def __init__(self, clip_model, num_ctx=4):
        super().__init__()
        self.clip_model = clip_model
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
        self.clip_model.text_model.embeddings.token_embedding.forward = custom_forward
        try:
            tf = self.clip_model.get_text_features(input_ids=input_ids)
        finally:
            self.clip_model.text_model.embeddings.token_embedding.forward = _ORIG_TOKEN_EMBEDDING_FORWARD
        return F.normalize(tf, dim=-1)

# Texts for Base CoOp vs Gemma-CoOp
texts_base = [f"{c}" for c in classes]
texts_gemma = [f"{c}, dog, recognized by {attributes[c]['attributes']['distinguishing_mark']}" for c in classes]

ids_base = make_coop_ids(texts_base, num_ctx=4)
ids_gemma = make_coop_ids(texts_gemma, num_ctx=4)

def train_coop_dogs(coop_ids, name="CoOp", epochs=50, lr=0.005, seed=42):
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

print("\n--- Training Standard CoOp on Stanford Dogs ---")
model_base, val_acc_base, ep_base = train_coop_dogs(ids_base, name="Baseline CoOp", epochs=50, lr=0.005)

print("\n--- Training Gemma-CoOp on Stanford Dogs ---")
model_gemma, val_acc_gemma, ep_gemma = train_coop_dogs(ids_gemma, name="Gemma-CoOp", epochs=50, lr=0.005)

# Evaluate on Test Set
with torch.no_grad():
    tf_base = model_base.get_text_features(ids_base)
    tf_gemma = model_gemma.get_text_features(ids_gemma)
    
    logits_base_test = (test_clip @ tf_base.t()) * scale
    m_base = compute_metrics(logits_base_test, test_labels)
    
    logits_gemma_test = (test_clip @ tf_gemma.t()) * scale
    m_gemma = compute_metrics(logits_gemma_test, test_labels)
    
    # Gemma-DualStream: grid search beta on Validation Set
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
    m_dual = compute_metrics(logits_dual_test, test_labels)

print("\n=== RESULTS ON STANFORD DOGS TEST SET (6,174 IMAGES) ===")
print(f"1. Baseline CoOp (2,048 params): {m_base['top1_acc']:.2f}% ({m_base['correct_count']}/6174) | Top-5: {m_base['top5_acc']:.2f}%")
print(f"2. Gemma-CoOp (2,048 params):    {m_gemma['top1_acc']:.2f}% ({m_gemma['correct_count']}/6174) | Top-5: {m_gemma['top5_acc']:.2f}%")
print(f"3. Gemma-DualStream (2,048 p.):  {m_dual['top1_acc']:.2f}% ({m_dual['correct_count']}/6174) | Top-5: {m_dual['top5_acc']:.2f}%")

# Save results to JSON
results = {
    "dataset": "Stanford Dogs",
    "protocol": "Stratified 90:10 train/val partition (13,024 train, 1,382 val). Checkpoint selected strictly on peak validation accuracy.",
    "models": {
        "coop_baseline": {
            "val_acc": val_acc_base,
            "best_epoch": ep_base,
            "test_top1": m_base["top1_acc"],
            "test_top5": m_base["top5_acc"],
            "correct_test": m_base["correct_count"],
            "total_test": len(test_labels),
            "trainable_params": 2048
        },
        "gemma_coop": {
            "val_acc": val_acc_gemma,
            "best_epoch": ep_gemma,
            "test_top1": m_gemma["top1_acc"],
            "test_top5": m_gemma["top5_acc"],
            "correct_test": m_gemma["correct_count"],
            "total_test": len(test_labels),
            "trainable_params": 2048
        },
        "gemma_dualstream": {
            "best_beta_on_val": best_beta,
            "val_acc": best_val_dual,
            "test_top1": m_dual["top1_acc"],
            "test_top5": m_dual["top5_acc"],
            "correct_test": m_dual["correct_count"],
            "total_test": len(test_labels),
            "trainable_params": 2048
        }
    }
}

output_path = os.path.join(RESULTS_DIR, "verified_stanford_dogs_dualstream.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved verified results to: {output_path}")
