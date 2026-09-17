"""
export_resnet50_weights.py — Extract features and train ResNet-50 adapters.
Target environment: Kaggle
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
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

# --- TTA PREPROCESSING ---
def apply_denoise(pil_img):
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    denoised_bgr = cv2.bilateralFilter(img_bgr, d=7, sigmaColor=50, sigmaSpace=50)
    return Image.fromarray(cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB))

def apply_clahe(pil_img):
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    clahe_bgr = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB))

def optimal_preprocessing_pipeline(pil_img):
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    denoised_bgr = cv2.bilateralFilter(img_bgr, d=7, sigmaColor=50, sigmaSpace=50)
    lab = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    clahe_bgr = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB))

def transform_geom(img):
    return optimal_preprocessing_pipeline(img).transpose(Image.FLIP_LEFT_RIGHT).rotate(15)

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Dataset root: {DATASET_ROOT}")
    
    print("\n[1] Loading dataset...")
    pets_trainval = OxfordIIITPet(root=str(DATASET_ROOT), split="trainval", download=True)
    
    cached_res_tta = KAGGLE_INPUT_DIR / "trainval_features_resnet50_tta.pt"
    cached_res_raw = KAGGLE_INPUT_DIR / "trainval_features_resnet50_raw.pt"
    cached_clip_lbls = KAGGLE_INPUT_DIR / "trainval_labels.pt"
    
    # We load labels from CLIP's export or directly extract them if missing
    if cached_clip_lbls.exists():
        trainval_labels = torch.load(cached_clip_lbls)
    else:
        trainval_labels = torch.tensor([lbl for _, lbl in pets_trainval])
    
    if cached_res_tta.exists() and cached_res_raw.exists():
        print("\n[2] Found cached ResNet-50 features! Loading directly (Skipping extraction)...")
        trainval_features_resnet = torch.load(cached_res_tta)
        f_raw = torch.load(cached_res_raw)
    else:
        print("\n[2] Loading ResNet-50...")
        res_weights = models.ResNet50_Weights.DEFAULT
        res_preprocess = res_weights.transforms()
        resnet_full = models.resnet50(weights=res_weights)
        backbone = nn.Sequential(*list(resnet_full.children())[:-1]).to(DEVICE)
        backbone.eval()
        
        class ResNetTransformDataset(Dataset):
            def __init__(self, base_ds, transform_fn=None):
                self.base_ds = base_ds
                self.transform_fn = transform_fn
            def __len__(self): return len(self.base_ds)
            def __getitem__(self, idx):
                img, lbl = self.base_ds[idx]
                if self.transform_fn: img = self.transform_fn(img)
                return res_preprocess(img), lbl
                
        def extract_res_feats(t_fn, desc):
            dl = DataLoader(ResNetTransformDataset(pets_trainval, t_fn), batch_size=128, shuffle=False)
            feats = []
            with torch.no_grad():
                for x, _ in tqdm(dl, desc=desc):
                    f = backbone(x.to(DEVICE)).squeeze(-1).squeeze(-1)
                    feats.append(f.cpu())
            return torch.cat(feats, dim=0)

        print("Extracting ResNet-50 5-view TTA Features...")
        f_raw = extract_res_feats(None, "View 1: Raw")
        f_denoise = extract_res_feats(apply_denoise, "View 2: Denoise")
        f_clahe = extract_res_feats(apply_clahe, "View 3: CLAHE")
        f_optimal = extract_res_feats(optimal_preprocessing_pipeline, "View 4: Optimal")
        f_geom = extract_res_feats(transform_geom, "View 5: Geometric")
        
        trainval_features_resnet = torch.stack([f_raw, f_denoise, f_clahe, f_optimal, f_geom], dim=0).mean(dim=0)
        
        torch.save(trainval_features_resnet, WEIGHTS_DIR / "trainval_features_resnet50_tta.pt")
        torch.save(f_raw, WEIGHTS_DIR / "trainval_features_resnet50_raw.pt")
    
    print("\n[3] Training ResNet-50 Adapters...")
    res_dl = DataLoader(TensorDataset(trainval_features_resnet.to(DEVICE), trainval_labels.to(DEVICE)), batch_size=128, shuffle=True)
    
    # 3.1 ResNet-50 Baseline (Raw features, no TTA)
    print("  Training ResNet-50 Baseline (Raw Image, No TTA)...")
    baseline_model = nn.Linear(2048, NUM_CLASSES).to(DEVICE)
    opt = torch.optim.AdamW(baseline_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    baseline_dl = DataLoader(TensorDataset(f_raw.to(DEVICE), trainval_labels.to(DEVICE)), batch_size=128, shuffle=True)
    for _ in range(30):
        for x, y in baseline_dl:
            opt.zero_grad()
            loss = F.cross_entropy(baseline_model(x), y)
            loss.backward()
            opt.step()
        scheduler.step()
    torch.save(baseline_model.state_dict(), WEIGHTS_DIR / "resnet50_baseline.pt")
    print("  -> Exported resnet50_baseline.pt")

    # 3.2 ResNet-50 Prototype
    prototypes = torch.zeros(NUM_CLASSES, 2048, device=DEVICE)
    class_counts = torch.zeros(NUM_CLASSES, device=DEVICE)
    # L2 normalize features before averaging
    train_norm = trainval_features_resnet.to(DEVICE) / trainval_features_resnet.to(DEVICE).norm(dim=-1, keepdim=True)
    for x, y in zip(train_norm, trainval_labels.to(DEVICE)):
        prototypes[y] += x
        class_counts[y] += 1
    prototypes = prototypes / class_counts.unsqueeze(1)
    # L2 normalize prototypes after averaging
    prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)
    torch.save(prototypes.cpu(), WEIGHTS_DIR / "resnet50_prototype.pt")
    print("  -> Exported resnet50_prototype.pt")

    # 3.3 ResNet-50 Linear Probe (TTA Features)
    linear_model = nn.Linear(2048, NUM_CLASSES).to(DEVICE)
    opt = torch.optim.AdamW(linear_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    for _ in range(30):
        for x, y in res_dl:
            opt.zero_grad()
            loss = F.cross_entropy(linear_model(x), y)
            loss.backward()
            opt.step()
        scheduler.step()
    torch.save(linear_model.state_dict(), WEIGHTS_DIR / "resnet50_linear.pt")
    print("  -> Exported resnet50_linear.pt")

    # 3.4 ResNet-50 Residual Adapter
    class ResNetAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = nn.Sequential(
                nn.Linear(2048, 512, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(512, 2048, bias=False),
                nn.ReLU(inplace=True)
            )
            self.fc = nn.Linear(2048, NUM_CLASSES)
            self.alpha = 0.2
        def forward(self, x):
            x = x + self.alpha * self.adapter(x)
            return self.fc(x)
            
    res_adapter = ResNetAdapter().to(DEVICE)
    opt = torch.optim.AdamW(res_adapter.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    for _ in range(30):
        for x, y in res_dl:
            opt.zero_grad()
            loss = F.cross_entropy(res_adapter(x), y)
            loss.backward()
            opt.step()
        scheduler.step()
    torch.save(res_adapter.state_dict(), WEIGHTS_DIR / "resnet50_adapter.pt")
    print("  -> Exported resnet50_adapter.pt")

    # 3.5 ResNet-50 Cosine Head
    class CosineClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(NUM_CLASSES, 2048))
            nn.init.xavier_uniform_(self.weight)
            self.tau = nn.Parameter(torch.tensor(10.0))
        def forward(self, x):
            x_norm = F.normalize(x, dim=-1)
            w_norm = F.normalize(self.weight, dim=-1)
            return self.tau * (x_norm @ w_norm.t())
            
    cosine_model = CosineClassifier().to(DEVICE)
    opt = torch.optim.AdamW(cosine_model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    for _ in range(30):
        for x, y in res_dl:
            opt.zero_grad()
            loss = F.cross_entropy(cosine_model(x), y)
            loss.backward()
            opt.step()
        scheduler.step()
    torch.save(cosine_model.state_dict(), WEIGHTS_DIR / "resnet50_cosine_full.pt")
    print("  -> Exported resnet50_cosine_full.pt")
    
    print("\n✅ EXPORT RESNET-50 COMPLETE! All weights saved to:", WEIGHTS_DIR)
