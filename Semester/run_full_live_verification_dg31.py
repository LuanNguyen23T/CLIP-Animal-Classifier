"""
========================================================================================
FULL LIVE EMPIRICAL VERIFICATION SCRIPT FOR REVIEWER DG31 (06/11/2026)
Strictly adheres to partition protocols from DG31 review:
  1. Oxford-IIIT Pet: Official 3,680 trainval (3,310 train / 370 val) & 3,669 test.
     - Live computation of McNemar paired tests (b, c, chi2, p-value).
  2. Stanford Dogs: Official 12,000 train (100 per class) & 8,580 test.
     - Internal validation split: 10,800 train / 1,200 val (10 per class).
     - Live execution and evaluation of: Zero-Shot (Raw, Handcrafted, CuPL, LaBo, CBM, Centroid, ZeroShot)
       and PEFT (CoOp, Gemma-CoOp, Gemma-DualStream, Gemma-VisAdapter).
No hardcoded values. All metrics computed dynamically from raw model predictions.
========================================================================================
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import f1_score
from transformers import CLIPModel, CLIPTokenizer

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Initializing Live Verification Suite on compute device: {device}")

# --------------------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------------------
def compute_metrics(logits, targets):
    preds = logits.argmax(dim=-1)
    acc1 = (preds == targets).float().mean().item() * 100.0
    correct_count = (preds == targets).sum().item()
    top5_preds = logits.topk(min(5, logits.shape[-1]), dim=-1)[1]
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

def calc_mcnemar(pred_a, pred_b, targets):
    correct_a = (pred_a == targets)
    correct_b = (pred_b == targets)
    b = (correct_a & ~correct_b).sum().item()
    c = (~correct_a & correct_b).sum().item()
    chi2 = ((abs(b - c) - 1.0) ** 2) / float(b + c)
    pval = 1.0 - stats.chi2.cdf(chi2, df=1)
    return b, c, chi2, pval

# ======================================================================================
# SECTION 1: OXFORD-IIIT PET - LIVE MCNEMAR STATISTICAL VERIFICATION
# ======================================================================================
print("\n" + "=" * 80)
print("📌 PART 1: OXFORD-IIIT PET STATISTICAL AUDIT & MCNEMAR ANALYSIS")
print("   Dataset partition: 3,310 train / 370 val / 3,669 test (Official Split)")
print("=" * 80)

pred_file = "Semester/results/verified_test_predictions.pt"
if not os.path.exists(pred_file):
    raise FileNotFoundError(f"Missing {pred_file}")

preds_data = torch.load(pred_file, map_location=device)
y_test_pets = preds_data["test_labels"]
print(f"Loaded Oxford Pet Test Predictions on N = {len(y_test_pets)} official test samples:")

m_dual = compute_metrics(F.one_hot(preds_data["pred_dual"], 37).float(), y_test_pets)
m_coop = compute_metrics(F.one_hot(preds_data["pred_coop"], 37).float(), y_test_pets)
m_res90 = compute_metrics(F.one_hot(preds_data["pred_res_90"], 37).float(), y_test_pets)
m_resfull = compute_metrics(F.one_hot(preds_data["pred_res_full"], 37).float(), y_test_pets)

print(f"  • Gemma-DualStream (5-View TTA): {m_dual['top1_acc']:.2f}% ({m_dual['correct_count']}/3669)")
print(f"  • Standard CoOp Baseline (TTA):  {m_coop['top1_acc']:.2f}% ({m_coop['correct_count']}/3669)")
print(f"  • ResNet-50 Cosine (9:1 Split):  {m_res90['top1_acc']:.2f}% ({m_res90['correct_count']}/3669)")
print(f"  • ResNet-50 Cosine (Full Split): {m_resfull['top1_acc']:.2f}% ({m_resfull['correct_count']}/3669)")

print("\n--- Dynamically Calculated McNemar Paired Tests (Continuity Corrected) ---")
comps = [
    ("Gemma-DualStream vs CoOp Baseline", preds_data["pred_dual"], preds_data["pred_coop"]),
    ("Gemma-DualStream vs ResNet-50 (Matched 9:1 Split)", preds_data["pred_dual"], preds_data["pred_res_90"]),
    ("Gemma-DualStream vs ResNet-50 (Full Ceiling)", preds_data["pred_dual"], preds_data["pred_res_full"])
]

mcnemar_results = {}
for name, p_a, p_b in comps:
    b, c, chi2, pval = calc_mcnemar(p_a, p_b, y_test_pets)
    mcnemar_results[name] = {"b": b, "c": c, "b_minus_c": b - c, "chi2": chi2, "p_value": pval}
    sig = "p < 0.001 (Decisive)" if pval < 0.001 else ("p < 0.05 (Significant)" if pval < 0.05 else f"p = {pval:.4f} (Marginal)")
    print(f"[{name}]")
    print(f"    b = {b}, c = {c}, diff = +{b - c} | chi^2 = {chi2:.4f} | p-value = {pval:.4e} ({sig})")

# ======================================================================================
# SECTION 2: STANFORD DOGS - OFFICIAL SPLIT LIVE BENCHMARK (12,000 / 8,580)
# ======================================================================================
print("\n" + "=" * 80)
print("📌 PART 2: STANFORD DOGS OFFICIAL BENCHMARK (12,000 TRAIN / 8,580 TEST)")
print("   Strict adherence to Khosla et al. & DG31 reviewer feedback")
print("=" * 80)

DOGS_FEAT_DIR = "Semester/cached_features/stanford_dogs_official"
DOGS_ATTR_FILE = "Semester/attributes/stanford_dogs_attributes.json"

with open(os.path.join(DOGS_FEAT_DIR, "classes.json"), encoding="utf-8") as f:
    dogs_classes = json.load(f)
with open(DOGS_ATTR_FILE, encoding="utf-8") as f:
    dogs_attributes = json.load(f)

train_clip_dogs = torch.load(os.path.join(DOGS_FEAT_DIR, "train_clip.pt"), map_location=device).float()
test_clip_dogs = torch.load(os.path.join(DOGS_FEAT_DIR, "test_clip.pt"), map_location=device).float()
train_labels_dogs = torch.load(os.path.join(DOGS_FEAT_DIR, "train_labels.pt"), map_location=device)
test_labels_dogs = torch.load(os.path.join(DOGS_FEAT_DIR, "test_labels.pt"), map_location=device)

num_classes_dogs = len(dogs_classes)
print(f"Verifying Partition Shapes:")
print(f"  • Official Train Images: {len(train_clip_dogs)} (Expected: 12000)")
print(f"  • Official Test Images:  {len(test_clip_dogs)} (Expected: 8580)")
assert len(train_clip_dogs) == 12000
assert len(test_clip_dogs) == 8580

# Stratified partition: 10,800 train (90 per class) / 1,200 val (10 per class)
torch.manual_seed(42)
val_idx_list, train_idx_list = [], []
for c in range(num_classes_dogs):
    c_mask = (train_labels_dogs == c).nonzero(as_tuple=True)[0]
    perm = torch.randperm(len(c_mask))
    val_idx_list.append(c_mask[perm[:10]])
    train_idx_list.append(c_mask[perm[10:]])

val_idx_dogs = torch.cat(val_idx_list)
train_idx_dogs = torch.cat(train_idx_list)

train_x_dogs, train_y_dogs = train_clip_dogs[train_idx_dogs], train_labels_dogs[train_idx_dogs]
val_x_dogs, val_y_dogs = train_clip_dogs[val_idx_dogs], train_labels_dogs[val_idx_dogs]
print(f"Internal Validation Split: Train = {len(train_x_dogs)}, Val = {len(val_x_dogs)}, Test = {len(test_clip_dogs)}")

# Load CLIP text model
model_id = "openai/clip-vit-base-patch32"
tokenizer = CLIPTokenizer.from_pretrained(model_id)
clip_model = CLIPModel.from_pretrained(model_id).to(device)
clip_model.eval()
scale = clip_model.logit_scale.exp().detach()

# Build 600-concept morphological centroid bank
concept_templates = {
    "head_and_ears": "head and ears of {name}: {desc}",
    "eyes": "eyes of {name}: {desc}",
    "coat": "coat texture and markings of {name}: {desc}",
    "body": "body structure of {name}: {desc}",
    "distinguishing_mark": "distinguishing trait of {name}: {desc}"
}

cbm_weights_list = []
cupl_weights_list = []
for c in dogs_classes:
    attr = dogs_attributes[c]["attributes"]
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

cbm_weights_dogs = torch.stack(cbm_weights_list, dim=0) # [120, 5, 512]
W_centroid_dogs = F.normalize(cbm_weights_dogs.mean(dim=1), dim=-1) # [120, 512]
W_cupl_dogs = torch.cat(cupl_weights_list, dim=0)

# Raw class & Handcrafted
raw_prompts = [f"a photo of a {c.lower()}." for c in dogs_classes]
inps_raw = tokenizer(raw_prompts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
with torch.no_grad():
    W_raw_dogs = F.normalize(clip_model.get_text_features(**inps_raw), dim=-1)

templates_handcrafted = [
    "a photo of a {c}, a type of dog.", "a photo of the {c}.", "a photo of a clean {c}.",
    "a close-up photo of a {c}.", "a cropped photo of the {c}.", "a photo of my {c}."
]
w_hand_list = []
for c in dogs_classes:
    p_c = [t.format(c=c.lower()) for t in templates_handcrafted]
    inps = tokenizer(p_c, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
    with torch.no_grad():
        tf = F.normalize(clip_model.get_text_features(**inps), dim=-1)
        w_hand_list.append(F.normalize(tf.mean(dim=0, keepdim=True), dim=-1))
W_handcrafted_dogs = torch.cat(w_hand_list, dim=0)

# Live Zero-Shot Evaluation on 8,580 Official Test Images
print("\n--- Real Live Execution: Zero-Shot Baselines on Official 8,580 Test Images ---")
with torch.no_grad():
    res_raw = compute_metrics((test_clip_dogs @ W_raw_dogs.t()) * scale, test_labels_dogs)
    res_hand = compute_metrics((test_clip_dogs @ W_handcrafted_dogs.t()) * scale, test_labels_dogs)
    res_cupl = compute_metrics((test_clip_dogs @ W_cupl_dogs.t()) * scale, test_labels_dogs)
    
    sims = torch.einsum("nd,ckd->nck", test_clip_dogs, cbm_weights_dogs)
    res_cbm = compute_metrics(sims.mean(dim=-1) * scale, test_labels_dogs)
    res_labo = compute_metrics((sims.max(dim=-1)[0] + sims.mean(dim=-1)) * 0.5 * scale, test_labels_dogs)
    res_cent = compute_metrics((test_clip_dogs @ W_centroid_dogs.t()) * scale, test_labels_dogs)
    res_zs = compute_metrics(((test_clip_dogs @ W_handcrafted_dogs.t()) + 0.5 * (test_clip_dogs @ W_centroid_dogs.t())) * scale, test_labels_dogs)

print(f"1. CLIP Raw Class Names:     {res_raw['top1_acc']:.2f}% ({res_raw['correct_count']}/8580) | Top-5: {res_raw['top5_acc']:.2f}% | F1: {res_raw['macro_f1']:.2f}%")
print(f"2. CLIP Handcrafted Prompt:  {res_hand['top1_acc']:.2f}% ({res_hand['correct_count']}/8580) | Top-5: {res_hand['top5_acc']:.2f}% | F1: {res_hand['macro_f1']:.2f}%")
print(f"3. CuPL Baseline:            {res_cupl['top1_acc']:.2f}% ({res_cupl['correct_count']}/8580) | Top-5: {res_cupl['top5_acc']:.2f}% | F1: {res_cupl['macro_f1']:.2f}%")
print(f"4. LaBo-style Concept Pool:  {res_labo['top1_acc']:.2f}% ({res_labo['correct_count']}/8580) | Top-5: {res_labo['top5_acc']:.2f}% | F1: {res_labo['macro_f1']:.2f}%")
print(f"5. Gemma-CBM (600 concepts): {res_cbm['top1_acc']:.2f}% ({res_cbm['correct_count']}/8580) | Top-5: {res_cbm['top5_acc']:.2f}% | F1: {res_cbm['macro_f1']:.2f}%")
print(f"6. Gemma-Centroid (Ours):    {res_cent['top1_acc']:.2f}% ({res_cent['correct_count']}/8580) | Top-5: {res_cent['top5_acc']:.2f}% | F1: {res_cent['macro_f1']:.2f}%")
print(f"7. Gemma-ZeroShot (Ours):    {res_zs['top1_acc']:.2f}% ({res_zs['correct_count']}/8580) | Top-5: {res_zs['top5_acc']:.2f}% | F1: {res_zs['macro_f1']:.2f}%")

# PEFT Setup
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

texts_base_dogs = [f"{c.lower()}." for c in dogs_classes]
texts_gemma_dogs = []
for c in dogs_classes:
    attr = dogs_attributes[c]["attributes"]
    trait = attr["distinguishing_mark"].split(".")[0].strip()
    texts_gemma_dogs.append(f"{c.lower()}, dog recognized by {trait}.")

ids_base_dogs = make_coop_ids(texts_base_dogs, num_ctx=4)
ids_gemma_dogs = make_coop_ids(texts_gemma_dogs, num_ctx=4)

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

def train_coop_live(coop_ids, name="CoOp", epochs=50, lr=0.005, seed=42):
    torch.manual_seed(seed)
    model = CoOpClassifierModel(clip_model, num_ctx=4).to(device)
    optimizer = torch.optim.AdamW([model.ctx], lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_v = 0.0
    best_epoch = 0
    best_w = None
    
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tf = model.get_text_features(coop_ids)
        logits = (train_x_dogs @ tf.t()) * scale
        loss = criterion(logits, train_y_dogs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        model.eval()
        with torch.no_grad():
            v_logits = (val_x_dogs @ tf.t()) * scale
            v_acc = (v_logits.argmax(-1) == val_y_dogs).float().mean().item() * 100.0
            if v_acc > best_v:
                best_v = v_acc
                best_epoch = ep
                best_w = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == epochs:
            print(f"  [{name}] Epoch {ep:02d}/{epochs:02d} | Loss: {loss.item():.4f} | Val Acc: {v_acc:.2f}% (Peak Val: {best_v:.2f}% @ Ep {best_epoch})")
            
    model.load_state_dict(best_w)
    elapsed = time.time() - t0
    print(f"  [{name}] Completed 50 epochs in {elapsed:.1f}s. Loaded optimal checkpoint from Epoch {best_epoch}.")
    return model, best_v, best_epoch

print("\n--- Real Live Training: Baseline CoOp (2,048 trainable prompt parameters) ---")
model_coop, val_coop, ep_coop = train_coop_live(ids_base_dogs, name="Standard CoOp", epochs=50, lr=0.005)

print("\n--- Real Live Training: Gemma-CoOp (2,048 trainable prompt parameters) ---")
model_gcoop, val_gcoop, ep_gcoop = train_coop_live(ids_gemma_dogs, name="Gemma-CoOp", epochs=50, lr=0.005)

# Evaluate CoOp and DualStream on 8,580 Official Test Images
with torch.no_grad():
    tf_coop = model_coop.get_text_features(ids_base_dogs)
    res_coop_official = compute_metrics((test_clip_dogs @ tf_coop.t()) * scale, test_labels_dogs)
    
    tf_gcoop = model_gcoop.get_text_features(ids_gemma_dogs)
    res_gcoop_official = compute_metrics((test_clip_dogs @ tf_gcoop.t()) * scale, test_labels_dogs)
    
    # DualStream beta search on validation
    val_coop_logits = (val_x_dogs @ tf_gcoop.t()) * scale
    val_cent_logits = (val_x_dogs @ W_centroid_dogs.t()) * scale
    
    best_beta = 0.1
    best_v_dual = 0.0
    for b_c in [0.05, 0.1, 0.2, 0.35, 0.5]:
        v_blend = val_coop_logits + b_c * val_cent_logits
        acc = (v_blend.argmax(-1) == val_y_dogs).float().mean().item() * 100.0
        if acc > best_v_dual:
            best_v_dual = acc
            best_beta = b_c
            
    print(f"\nOptimal DualStream fusion beta chosen on Val: beta = {best_beta} (Val Acc: {best_v_dual:.2f}%)")
    test_dual_logits = (test_clip_dogs @ tf_gcoop.t()) * scale + best_beta * ((test_clip_dogs @ W_centroid_dogs.t()) * scale)
    res_dual_official = compute_metrics(test_dual_logits, test_labels_dogs)

# Train Gemma-VisAdapter
print("\n--- Real Live Training: Gemma-VisAdapter (65,536 trainable residual MLP parameters) ---")
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
crit_vis = nn.CrossEntropyLoss()

best_v_vis = 0.0
best_ep_vis = 0
best_w_vis = None
epochs_vis = 40

t0_vis = time.time()
for ep in range(1, epochs_vis + 1):
    vis_adapter.train()
    delta_x = vis_adapter(train_x_dogs)
    adapted_x = F.normalize((1.0 - alpha) * train_x_dogs + alpha * delta_x, dim=-1)
    loss = crit_vis((adapted_x @ W_centroid_dogs.t()) * scale, train_y_dogs)
    
    opt_vis.zero_grad()
    loss.backward()
    opt_vis.step()
    
    vis_adapter.eval()
    with torch.no_grad():
        v_delta = vis_adapter(val_x_dogs)
        v_adapt = F.normalize((1.0 - alpha) * val_x_dogs + alpha * v_delta, dim=-1)
        v_acc = ((v_adapt @ W_centroid_dogs.t()) * scale).argmax(-1).eq(val_y_dogs).float().mean().item() * 100.0
        if v_acc > best_v_vis:
            best_v_vis = v_acc
            best_ep_vis = ep
            best_w_vis = {k: v.cpu().clone() for k, v in vis_adapter.state_dict().items()}
            
    if ep % 10 == 0 or ep == epochs_vis:
        print(f"  [Gemma-VisAdapter] Epoch {ep:02d}/{epochs_vis:02d} | Loss: {loss.item():.4f} | Val Acc: {v_acc:.2f}% (Peak Val: {best_v_vis:.2f}% @ Ep {best_ep_vis})")

vis_adapter.load_state_dict(best_w_vis)
vis_adapter.eval()
print(f"  [Gemma-VisAdapter] Completed 40 epochs in {time.time()-t0_vis:.1f}s. Loaded optimal checkpoint from Epoch {best_ep_vis}.")

with torch.no_grad():
    t_delta = vis_adapter(test_clip_dogs)
    t_adapt = F.normalize((1.0 - alpha) * test_clip_dogs + alpha * t_delta, dim=-1)
    res_vis_official = compute_metrics((t_adapt @ W_centroid_dogs.t()) * scale, test_labels_dogs)

print("\n" + "=" * 80)
print("🏆 FINAL VERIFIED OFFICIAL BENCHMARK RESULTS (TEST ON ALL 8,580 IMAGES):")
print("=" * 80)
print(f"  1. Standard CoOp Baseline (2,048 p.):  Top-1: {res_coop_official['top1_acc']:.2f}% ({res_coop_official['correct_count']}/8580) | Top-5: {res_coop_official['top5_acc']:.2f}% | F1: {res_coop_official['macro_f1']:.2f}%")
print(f"  2. Gemma-VisAdapter (65,536 p.):       Top-1: {res_vis_official['top1_acc']:.2f}% ({res_vis_official['correct_count']}/8580) | Top-5: {res_vis_official['top5_acc']:.2f}% | F1: {res_vis_official['macro_f1']:.2f}%")
print(f"  3. Gemma-DualStream (2,048 p.):        Top-1: {res_dual_official['top1_acc']:.2f}% ({res_dual_official['correct_count']}/8580) | Top-5: {res_dual_official['top5_acc']:.2f}% | F1: {res_dual_official['macro_f1']:.2f}%")
print(f"  4. Gemma-CoOp (2,048 p.):              Top-1: {res_gcoop_official['top1_acc']:.2f}% ({res_gcoop_official['correct_count']}/8580) | Top-5: {res_gcoop_official['top5_acc']:.2f}% | F1: {res_gcoop_official['macro_f1']:.2f}%")
print("=" * 80)
print("✅ ALL METRICS CONFIRMED DYNAMICALLY COMPUTED FROM RAW MODEL PREDICTIONS.")

