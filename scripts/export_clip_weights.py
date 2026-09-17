"""
export_clip_weights.py — Extract features and train CLIP models.
Target environment: Kaggle
"""
import os
import sys
from pathlib import Path

from transformers import CLIPModel, CLIPProcessor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.datasets import OxfordIIITPet
from tqdm.auto import tqdm

# --- CONFIG FOR KAGGLE ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
KAGGLE_INPUT_DIR = Path("/kaggle/input/datasets/vanton2005/allmodel")
DATASET_ROOT = Path("/kaggle/working/dataset")
WEIGHTS_DIR = Path("/kaggle/working/weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
DATASET_ROOT.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 37

PROMPT_TEMPLATES = [
    "Picture of a pet with breed: {c}, which is a type of animal."
]

OXFORD_CLASSES = [
    'Abyssinian', 'American Bulldog', 'American Pit Bull Terrier', 'Basset Hound', 
    'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer', 'British Shorthair', 
    'Chihuahua', 'Egyptian Mau', 'English Cocker Spaniel', 'English Setter', 
    'German Shorthaired', 'Great Pyrenees', 'Havanese', 'Japanese Chin', 
    'Keeshond', 'Leonberger', 'Maine Coon', 'Miniature Pinscher', 'Newfoundland', 
    'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian Blue', 'Saint Bernard', 
    'Samoyed', 'Scottish Terrier', 'Shiba Inu', 'Siamese', 'Sphynx', 
    'Staffordshire Bull Terrier', 'Wheaten Terrier', 'Yorkshire Terrier'
]

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

TEXT_PROMPTS = [PROMPT_TEMPLATES[0].format(c=c) for c in OXFORD_CLASSES]

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Dataset root: {DATASET_ROOT}")
    
    print("\n[1] Loading dataset...")
    pets_trainval = OxfordIIITPet(root=str(DATASET_ROOT), split="trainval", download=True)
    
    cached_clip_feats = KAGGLE_INPUT_DIR / "trainval_features.pt"
    cached_clip_lbls = KAGGLE_INPUT_DIR / "trainval_labels.pt"
    
    # ---------------------------------------------------------
    # BẮT BUỘC TẠO LẠI TEXT FEATURES ĐỂ TRÁNH LỖI CŨ (36 classes)
    # ---------------------------------------------------------
    print("Extracting CLIP Text Features (Forced rebuild)...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    text_inputs = clip_processor(text=TEXT_PROMPTS, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        out = clip_model.get_text_features(**text_inputs)
        if isinstance(out, torch.Tensor):
            text_features = out
        else:
            text_features = clip_model.text_projection(out.pooler_output)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_enr = text_features.cpu()
        torch.save(text_features_enr, WEIGHTS_DIR / "text_features_enr.pt")
        print(" -> Exported text_features_enr.pt with shape:", text_features_enr.shape)
    
    if cached_clip_feats.exists() and cached_clip_lbls.exists():
        print("\n[2] Found cached CLIP features! Loading directly (Skipping extraction)...")
        trainval_features_clip = torch.load(cached_clip_feats)
        trainval_labels = torch.load(cached_clip_lbls)
    else:
        print("\n[2] Loading CLIP Image Encoder...")
        class CLIPTransformDataset(Dataset):
            def __init__(self, base_ds, processor):
                self.base_ds = base_ds
                self.processor = processor
            def __len__(self): return len(self.base_ds)
            def __getitem__(self, idx):
                img, lbl = self.base_ds[idx]
                pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
                return pixel_values, lbl

        clip_loader = DataLoader(CLIPTransformDataset(pets_trainval, clip_processor), batch_size=128, shuffle=False)
        
        print("Extracting CLIP Features (Raw Preprocessing)...")
        clip_feats, clip_lbls = [], []
        with torch.no_grad():
            for x, y in tqdm(clip_loader, desc="CLIP Trainval"):
                x = x.to(DEVICE)
                out = clip_model.get_image_features(pixel_values=x)
                if isinstance(out, torch.Tensor):
                    f = out
                else:
                    f = clip_model.visual_projection(out.pooler_output)
                f = f / f.norm(dim=-1, keepdim=True)
                clip_feats.append(f.cpu())
                clip_lbls.append(y)
        trainval_features_clip = torch.cat(clip_feats, dim=0)
        trainval_labels = torch.cat(clip_lbls, dim=0)
        
        torch.save(trainval_features_clip, WEIGHTS_DIR / "trainval_features.pt")
        torch.save(trainval_labels, WEIGHTS_DIR / "trainval_labels.pt")

    logit_scale_val = clip_model.logit_scale.exp().item()
    print(f"Logit scale: {logit_scale_val}")

    print("\n[3] Training CLIP Adapters...")
    
    # 3.1 Tip-Adapter (Free)
    torch.save({"W_1": trainval_features_clip, "labels": trainval_labels}, WEIGHTS_DIR / "tip_adapter_cache.pt")
    print("  -> Exported tip_adapter_cache.pt")
    
    class TipAdapterF(nn.Module):
        def __init__(self, w1, labels, text_feats):
            super().__init__()
            self.W_1 = nn.Parameter(w1.clone())
            self.register_buffer("W_2", F.one_hot(labels, num_classes=NUM_CLASSES).float())
            self.register_buffer("W_text", text_feats.clone())
            self.alpha = 1.0
            self.beta = 10.0
            self.register_buffer("K_avg", torch.tensor(len(w1) / float(NUM_CLASSES)))

        def forward(self, x):
            x_norm = F.normalize(x, dim=-1)
            logits_zs = (x_norm @ self.W_text.t()) * logit_scale_val
            w1_norm = self.W_1 / self.W_1.norm(dim=-1, keepdim=True)
            sim_matrix = x_norm @ w1_norm.t()
            A = torch.exp(-self.beta * (1.0 - sim_matrix))
            logits_cache = ((A @ self.W_2) / self.K_avg) * logit_scale_val
            return logits_zs + self.alpha * logits_cache
            
    tip_model = TipAdapterF(trainval_features_clip.to(DEVICE), trainval_labels.to(DEVICE), text_features_enr.to(DEVICE)).to(DEVICE)
    opt = torch.optim.SGD([tip_model.W_1], lr=0.001, momentum=0.9, weight_decay=1e-4)
    clip_dl = DataLoader(TensorDataset(trainval_features_clip.to(DEVICE), trainval_labels.to(DEVICE)), batch_size=256, shuffle=True)
    for _ in range(20): # 20 epochs
        for x, y in clip_dl:
            opt.zero_grad()
            loss = F.cross_entropy(tip_model(x), y)
            loss.backward()
            opt.step()
    torch.save(tip_model.state_dict(), WEIGHTS_DIR / "tip_adapter_f_full.pt")
    print("  -> Exported tip_adapter_f_full.pt")

    class CLIPAdapter(nn.Module):
        def __init__(self, input_dim=512, bottleneck_dim=64, alpha=0.2):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, bottleneck_dim),
                nn.ReLU(),
                nn.Linear(bottleneck_dim, input_dim)
            )
            self.alpha = alpha
            nn.init.constant_(self.mlp[2].weight, 0.0)
            nn.init.constant_(self.mlp[2].bias, 0.0)
            
        def forward(self, x):
            res = self.mlp(x)
            x_adapted = self.alpha * res + (1.0 - self.alpha) * x
            return x_adapted / x_adapted.norm(dim=-1, keepdim=True)
            
    class AdapterClassifier(nn.Module):
        def __init__(self, adapter, text_features, logit_scale):
            super().__init__()
            self.adapter = adapter
            self.register_buffer("text_features", text_features.clone())
            self.logit_scale = logit_scale
            
        def forward(self, x):
            x_adapted = self.adapter(x)
            logits = (x_adapted @ self.text_features.t()) * self.logit_scale
            return logits

    ca_base = CLIPAdapter(input_dim=512, bottleneck_dim=64, alpha=0.2).to(DEVICE)
    ca_model = AdapterClassifier(ca_base, text_features_enr.to(DEVICE), logit_scale=logit_scale_val).to(DEVICE)
    
    opt = torch.optim.AdamW(ca_model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(50): # 50 epochs
        for x, y in clip_dl:
            opt.zero_grad()
            loss = F.cross_entropy(ca_model(x), y)
            loss.backward()
            opt.step()
    torch.save(ca_base.state_dict(), WEIGHTS_DIR / "clip_adapter_full.pt")
    print("  -> Exported clip_adapter_full.pt")
    
    print("\n✅ EXPORT CLIP COMPLETE! All weights saved to:", WEIGHTS_DIR)
