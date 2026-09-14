"""
export_coop_tip.py — Export only CoOp & Tip-Adapter-F (CLIP-Adapter already done).
Run from project root: python scripts/export_coop_tip.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import CLIPModel, CLIPProcessor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "openai/clip-vit-base-patch32"
NUM_CLASSES = 37
NUM_CTX = 4
WEIGHTS_DIR = ROOT / "backend" / "weights"

BREED_CATALOGUE = [
    (1,"Abyssinian","Cat","Abyssinian cat"),(2,"american_bulldog","Dog","American Bulldog"),
    (3,"american_pit_bull_terrier","Dog","American Pit Bull Terrier"),(4,"basset_hound","Dog","Basset Hound"),
    (5,"beagle","Dog","Beagle"),(6,"Bengal","Cat","Bengal cat"),(7,"Birman","Cat","Birman cat"),
    (8,"Bombay","Cat","Bombay cat"),(9,"boxer","Dog","Boxer dog"),(10,"British_Shorthair","Cat","British Shorthair cat"),
    (11,"chihuahua","Dog","Chihuahua"),(12,"Egyptian_Mau","Cat","Egyptian Mau cat"),
    (13,"english_cocker_spaniel","Dog","English Cocker Spaniel"),(14,"english_setter","Dog","English Setter"),
    (15,"german_shorthaired","Dog","German Shorthaired Pointer"),(16,"great_pyrenees","Dog","Great Pyrenees"),
    (17,"havanese","Dog","Havanese"),(18,"japanese_chin","Dog","Japanese Chin"),(19,"keeshond","Dog","Keeshond"),
    (20,"leonberger","Dog","Leonberger"),(21,"Maine_Coon","Cat","Maine Coon cat"),(22,"miniature_pinscher","Dog","Miniature Pinscher"),
    (23,"newfoundland","Dog","Newfoundland dog"),(24,"Persian","Cat","Persian cat"),(25,"pomeranian","Dog","Pomeranian"),
    (26,"pug","Dog","Pug"),(27,"Ragdoll","Cat","Ragdoll cat"),(28,"Russian_Blue","Cat","Russian Blue cat"),
    (29,"saint_bernard","Dog","Saint Bernard"),(30,"samoyed","Dog","Samoyed"),(31,"scottish_terrier","Dog","Scottish Terrier"),
    (32,"shiba_inu","Dog","Shiba Inu"),(33,"Siamese","Cat","Siamese cat"),(34,"Sphynx","Cat","Sphynx cat"),
    (35,"staffordshire_bull_terrier","Dog","Staffordshire Bull Terrier"),(36,"wheaten_terrier","Dog","Wheaten Terrier"),
    (37,"yorkshire_terrier","Dog","Yorkshire Terrier"),
]
CLASS_NAMES = [b[3] for b in BREED_CATALOGUE]

print(f"Device: {DEVICE}")

# Load model
print("\nLoading CLIP...")
clip_model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(MODEL_ID)
clip_model.eval()

# Load cached text features (already exported)
text_features_enr = torch.load(WEIGHTS_DIR / "text_features_enr.pt", map_location=DEVICE)
logit_scale_val = clip_model.logit_scale.exp().item()
print(f"  logit_scale = {logit_scale_val:.2f}")

# Load train features
print("Loading cached features...")
feat_path = ROOT / "dataset" / "cached" / "trainval_features_optimal.pt"
if not feat_path.exists():
    feat_path = ROOT / "dataset" / "trainval_features.pt"
train_feats = F.normalize(torch.load(feat_path, map_location="cpu").float(), dim=-1)
train_lbls = torch.load(ROOT / "dataset" / "trainval_labels.pt", map_location="cpu")
loader = DataLoader(TensorDataset(train_feats, train_lbls), batch_size=256, shuffle=True)
criterion = nn.CrossEntropyLoss()
print(f"  {train_feats.shape}, {train_lbls.shape}")

# ── CoOp ──────────────────────────────────────────────────────────────────────
print("\n[1/2] Training CoOp (50 epochs)...")

def make_coop_input_ids(class_names, processor, num_ctx=4):
    result = []
    for name in class_names:
        tokens = processor.tokenizer(name, return_tensors="pt")["input_ids"][0]
        sos, name_tokens, eos = tokens[:1], tokens[1:-1], tokens[-1:]
        ctx_tokens = torch.tensor([49400 + i for i in range(num_ctx)])
        seq = torch.cat([sos, ctx_tokens, name_tokens, eos])
        pad_len = 77 - len(seq)
        seq = torch.cat([seq, torch.zeros(pad_len, dtype=torch.long)]) if pad_len > 0 else seq[:77]
        if pad_len <= 0: seq[-1] = eos[0]
        result.append(seq)
    return torch.stack(result)

coop_input_ids = make_coop_input_ids(CLASS_NAMES, clip_processor, NUM_CTX).to(DEVICE)

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
        opt_coop.zero_grad(); criterion(model_coop(xb, coop_input_ids), yb).backward(); opt_coop.step()
    if (epoch + 1) % 10 == 0: print(f"  Epoch {epoch+1}/50")

torch.save(model_coop.ctx.data.cpu(), WEIGHTS_DIR / "coop_context_full.pt")
print(f"  Saved coop_context_full.pt — shape {model_coop.ctx.shape}")

# ── Tip-Adapter-F ─────────────────────────────────────────────────────────────
print("\n[2/2] Training Tip-Adapter-F (20 epochs)...")

class TipAdapterF(nn.Module):
    def __init__(self, supp_f, supp_l, text_f, alpha=1.0, beta=10.0):
        super().__init__()
        self.W_1 = nn.Parameter(supp_f.clone())
        self.register_buffer("W_2", F.one_hot(supp_l, num_classes=NUM_CLASSES).float())
        self.register_buffer("W_text", text_f.clone())
        self.alpha = alpha; self.beta = beta
        self.register_buffer("K_avg", torch.tensor(len(supp_f) / float(NUM_CLASSES)))
        self.logit_scale = logit_scale_val

    def forward(self, x):
        logits_zs = (x @ self.W_text.t()) * self.logit_scale
        w1n = self.W_1 / self.W_1.norm(dim=-1, keepdim=True)
        A = torch.exp(-self.beta * (1.0 - x @ w1n.t()))
        return logits_zs + self.alpha * ((A @ self.W_2) / self.K_avg) * self.logit_scale

model_tip = TipAdapterF(train_feats.to(DEVICE), train_lbls.to(DEVICE), text_features_enr).to(DEVICE)
opt_tip = torch.optim.SGD([model_tip.W_1], lr=1e-3, momentum=0.9, weight_decay=1e-4)

for epoch in range(20):
    model_tip.train()
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt_tip.zero_grad(); criterion(model_tip(xb), yb).backward(); opt_tip.step()
    if (epoch + 1) % 5 == 0: print(f"  Epoch {epoch+1}/20")

torch.save(model_tip.state_dict(), WEIGHTS_DIR / "tip_adapter_f_full.pt")
print("  Saved tip_adapter_f_full.pt")

print("\n" + "="*60)
print("ALL WEIGHTS EXPORTED")
print("="*60)
for f in sorted(WEIGHTS_DIR.iterdir()):
    print(f"  {f.name:45} {f.stat().st_size/1024:8.1f} KB")
