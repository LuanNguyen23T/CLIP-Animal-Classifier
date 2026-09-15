"""
export_weights.py — Train & export model weights for CLIP-Adapter, CoOp, Tip-Adapter-F.

Usage (from project root):
    python scripts/export_weights.py

Requirements:
    - dataset/cached/trainval_features_optimal.pt  (shape [3680, 512])
    - dataset/trainval_labels.pt                   (shape [3680])
    - torch, transformers installed

Output (backend/weights/):
    text_features_enr.pt      — Enriched text embeddings [37, 512]
    clip_adapter_full.pt      — CLIPAdapter + head state_dict
    coop_context_full.pt      — CoOp ctx tensor [4, 512]
    tip_adapter_f_full.pt     — TipAdapterF state_dict
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import CLIPModel, CLIPProcessor

# ─── Config ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
MODEL_ID = "openai/clip-vit-base-patch32"
NUM_CLASSES = 37
NUM_CTX = 4
WEIGHTS_DIR = ROOT / "backend" / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Output dir: {WEIGHTS_DIR}")

# ─── Breed catalogue ─────────────────────────────────────────────────────────
BREED_CATALOGUE = [
    (1,  "Abyssinian",                "Cat", "Abyssinian cat"),
    (2,  "american_bulldog",          "Dog", "American Bulldog"),
    (3,  "american_pit_bull_terrier", "Dog", "American Pit Bull Terrier"),
    (4,  "basset_hound",              "Dog", "Basset Hound"),
    (5,  "beagle",                    "Dog", "Beagle"),
    (6,  "Bengal",                    "Cat", "Bengal cat"),
    (7,  "Birman",                    "Cat", "Birman cat"),
    (8,  "Bombay",                    "Cat", "Bombay cat"),
    (9,  "boxer",                     "Dog", "Boxer dog"),
    (10, "British_Shorthair",         "Cat", "British Shorthair cat"),
    (11, "chihuahua",                 "Dog", "Chihuahua"),
    (12, "Egyptian_Mau",              "Cat", "Egyptian Mau cat"),
    (13, "english_cocker_spaniel",    "Dog", "English Cocker Spaniel"),
    (14, "english_setter",            "Dog", "English Setter"),
    (15, "german_shorthaired",        "Dog", "German Shorthaired Pointer"),
    (16, "great_pyrenees",            "Dog", "Great Pyrenees"),
    (17, "havanese",                  "Dog", "Havanese"),
    (18, "japanese_chin",             "Dog", "Japanese Chin"),
    (19, "keeshond",                  "Dog", "Keeshond"),
    (20, "leonberger",                "Dog", "Leonberger"),
    (21, "Maine_Coon",                "Cat", "Maine Coon cat"),
    (22, "miniature_pinscher",        "Dog", "Miniature Pinscher"),
    (23, "newfoundland",              "Dog", "Newfoundland dog"),
    (24, "Persian",                   "Cat", "Persian cat"),
    (25, "pomeranian",                "Dog", "Pomeranian"),
    (26, "pug",                       "Dog", "Pug"),
    (27, "Ragdoll",                   "Cat", "Ragdoll cat"),
    (28, "Russian_Blue",              "Cat", "Russian Blue cat"),
    (29, "saint_bernard",             "Dog", "Saint Bernard"),
    (30, "samoyed",                   "Dog", "Samoyed"),
    (31, "scottish_terrier",          "Dog", "Scottish Terrier"),
    (32, "shiba_inu",                 "Dog", "Shiba Inu"),
    (33, "Siamese",                   "Cat", "Siamese cat"),
    (34, "Sphynx",                    "Cat", "Sphynx cat"),
    (35, "staffordshire_bull_terrier","Dog", "Staffordshire Bull Terrier"),
    (36, "wheaten_terrier",           "Dog", "Wheaten Terrier"),
    (37, "yorkshire_terrier",         "Dog", "Yorkshire Terrier"),
]
CLASS_NAMES = [b[1] for b in BREED_CATALOGUE]

PROMPT_TEMPLATES = [
    "Picture of a pet with breed: {raw_name}, which is a type of animal."
]

# ─── 1. Load CLIP & build text features ──────────────────────────────────────
print("\n[1/5] Loading CLIP model...")
clip_model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(MODEL_ID)
clip_model.eval()
logit_scale_val = clip_model.logit_scale.exp().item()
print(f"  logit_scale = {logit_scale_val:.2f}")

print("[1/5] Building enriched text features...")
all_prompts = []
for _, raw_name, _, _ in BREED_CATALOGUE:
    for tmpl in PROMPT_TEMPLATES:
        all_prompts.append(tmpl.format(raw_name=raw_name))

inputs = clip_processor(text=all_prompts, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
with torch.no_grad():
    out = clip_model.get_text_features(**inputs)
    text_feats = out if isinstance(out, torch.Tensor) else out.pooler_output
    text_feats = F.normalize(text_feats, dim=-1)

text_feats = text_feats.view(NUM_CLASSES, len(PROMPT_TEMPLATES), -1).mean(dim=1)
text_features_enr = F.normalize(text_feats, dim=-1)

torch.save(text_features_enr.cpu(), WEIGHTS_DIR / "text_features_enr.pt")
print(f"  Saved text_features_enr.pt — shape: {text_features_enr.shape}")

# ─── 2. Load cached train features ───────────────────────────────────────────
print("\n[2/5] Loading cached train features...")
feat_path = ROOT / "dataset" / "cached" / "trainval_features_optimal.pt"
if not feat_path.exists():
    feat_path = ROOT / "dataset" / "trainval_features.pt"
lbl_path = ROOT / "dataset" / "trainval_labels.pt"

train_feats = F.normalize(torch.load(feat_path, map_location="cpu").float(), dim=-1)
train_lbls  = torch.load(lbl_path, map_location="cpu")
print(f"  train_feats: {train_feats.shape}, train_lbls: {train_lbls.shape}")
loader = DataLoader(TensorDataset(train_feats, train_lbls), batch_size=256, shuffle=True)
criterion = nn.CrossEntropyLoss()

# ─── 3. CLIP-Adapter ─────────────────────────────────────────────────────────
print("\n[3/5] Training CLIP-Adapter (50 epochs)...")

class CLIPAdapter(nn.Module):
    def __init__(self, input_dim=512, bottleneck_dim=64, alpha=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim), nn.ReLU(),
            nn.Linear(bottleneck_dim, input_dim)
        )
        self.alpha = alpha
        nn.init.constant_(self.mlp[2].weight, 0.0)
        nn.init.constant_(self.mlp[2].bias, 0.0)

    def forward(self, x):
        res = self.mlp(x)
        x_a = self.alpha * res + (1.0 - self.alpha) * x
        return x_a / x_a.norm(dim=-1, keepdim=True)

class AdapterClassifier(nn.Module):
    def __init__(self, adapter, text_features, logit_scale=100.0):
        super().__init__()
        self.adapter = adapter
        self.register_buffer("text_features", text_features.clone())
        self.logit_scale = logit_scale

    def forward(self, x):
        return (self.adapter(x) @ self.text_features.t()) * self.logit_scale

adapter = CLIPAdapter().to(DEVICE)
model_ca = AdapterClassifier(adapter, text_features_enr.to(DEVICE), logit_scale_val).to(DEVICE)
opt = torch.optim.AdamW(model_ca.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(50):
    model_ca.train()
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        criterion(model_ca(xb), yb).backward()
        opt.step()
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/50")

torch.save(model_ca.state_dict(), WEIGHTS_DIR / "clip_adapter_full.pt")
print("  Saved clip_adapter_full.pt")

# ─── 4. CoOp ─────────────────────────────────────────────────────────────────
print("\n[4/5] Training CoOp Prompt Tuning (50 epochs)...")

def construct_coop_input_ids(class_names, processor, num_ctx=4):
    result = []
    for name in class_names:
        tokens = processor.tokenizer(name, return_tensors="pt")["input_ids"][0]
        sos, name_tokens, eos = tokens[:1], tokens[1:-1], tokens[-1:]
        ctx_tokens = torch.tensor([49400 + i for i in range(num_ctx)])
        seq = torch.cat([sos, ctx_tokens, name_tokens, eos])
        pad_len = 77 - len(seq)
        if pad_len > 0:
            seq = torch.cat([seq, torch.zeros(pad_len, dtype=torch.long)])
        else:
            seq = seq[:77]; seq[-1] = eos[0]
        result.append(seq)
    return torch.stack(result)

coop_input_ids = construct_coop_input_ids(CLASS_NAMES, clip_processor, NUM_CTX).to(DEVICE)

class CoOpClassifier(nn.Module):
    def __init__(self, clip_model, num_ctx=4):
        super().__init__()
        self.clip_model = clip_model
        self.ctx = nn.Parameter(torch.randn(num_ctx, 512, device=DEVICE) * 0.02)
        orig_fwd = clip_model.text_model.embeddings.token_embedding.forward
        def custom_fwd(input_ids, *a, **kw):
            emb = orig_fwd(input_ids, *a, **kw).clone()
            for i in range(num_ctx):
                mask = (input_ids == (49400 + i))
                if mask.any(): emb[mask] = self.ctx[i].to(emb.dtype)
            return emb
        clip_model.text_model.embeddings.token_embedding.forward = custom_fwd

    def forward(self, img_feats, input_ids):
        tf = self.clip_model.get_text_features(input_ids=input_ids)
        tf = tf if isinstance(tf, torch.Tensor) else tf.pooler_output
        tf = tf / tf.norm(dim=-1, keepdim=True)
        return (img_feats @ tf.t()) * self.clip_model.logit_scale.exp()

model_coop = CoOpClassifier(clip_model, NUM_CTX).to(DEVICE)
opt_coop = torch.optim.AdamW([model_coop.ctx], lr=2e-3, weight_decay=1e-4)

for epoch in range(50):
    model_coop.train()
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt_coop.zero_grad()
        criterion(model_coop(xb, coop_input_ids), yb).backward()
        opt_coop.step()
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/50")

torch.save(model_coop.ctx.data.cpu(), WEIGHTS_DIR / "coop_context_full.pt")
print(f"  Saved coop_context_full.pt — shape: {model_coop.ctx.shape}")

# ─── 5. Tip-Adapter-F ────────────────────────────────────────────────────────
print("\n[5/5] Training Tip-Adapter-F (20 epochs)...")

class TipAdapterF(nn.Module):
    def __init__(self, support_features, support_labels, text_features, alpha=1.0, beta=10.0):
        super().__init__()
        self.W_1 = nn.Parameter(support_features.clone())
        self.register_buffer("W_2", F.one_hot(support_labels, num_classes=NUM_CLASSES).float())
        self.register_buffer("W_text", text_features.clone())
        self.alpha = alpha; self.beta = beta
        self.register_buffer("K_avg", torch.tensor(len(support_features) / float(NUM_CLASSES)))
        self.logit_scale = logit_scale_val

    def forward(self, x):
        logits_zs = (x @ self.W_text.t()) * self.logit_scale
        w1n = self.W_1 / self.W_1.norm(dim=-1, keepdim=True)
        A = torch.exp(-self.beta * (1.0 - x @ w1n.t()))
        return logits_zs + self.alpha * ((A @ self.W_2) / self.K_avg) * self.logit_scale

model_tip = TipAdapterF(train_feats.to(DEVICE), train_lbls.to(DEVICE),
                        text_features_enr.to(DEVICE)).to(DEVICE)
opt_tip = torch.optim.SGD([model_tip.W_1], lr=1e-3, momentum=0.9, weight_decay=1e-4)

for epoch in range(20):
    model_tip.train()
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt_tip.zero_grad()
        criterion(model_tip(xb), yb).backward()
        opt_tip.step()
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}/20")

torch.save(model_tip.state_dict(), WEIGHTS_DIR / "tip_adapter_f_full.pt")
print("  Saved tip_adapter_f_full.pt")

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EXPORT COMPLETE")
print("="*60)
for f in sorted(WEIGHTS_DIR.iterdir()):
    print(f"  {f.name:45} {f.stat().st_size/1024:8.1f} KB")
print("\nAll weights ready. You can now start the backend server.")
