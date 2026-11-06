"""
Extract and cache normalized CLIP ViT-B/32 visual features for Stanford Dogs
using the official benchmark split (12,000 train / 8,580 test images).
"""

import os
import json
import time
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Feature extraction compute device: {device}")

SAVE_DIR = "Semester/cached_features/stanford_dogs_official"
os.makedirs(SAVE_DIR, exist_ok=True)
ATTR_FILE = "Semester/attributes/stanford_dogs_attributes.json"

with open(ATTR_FILE, encoding="utf-8") as f:
    attributes = json.load(f)

clean_to_attr = {k.replace("-", " ").lower(): k for k in attributes.keys()}

print("Loading official Stanford Dogs dataset (maurice-fp/stanford-dogs)...")
ds = load_dataset("maurice-fp/stanford-dogs")
train_ds = ds["train"]
test_ds = ds["test"]

raw_class_names = train_ds.features["label"].names
classes_aligned = []
for raw in raw_class_names:
    sub = raw.split("-", 1)[1] if "-" in raw else raw
    key = sub.replace("_", " ").replace("-", " ").strip().lower()
    classes_aligned.append(clean_to_attr[key])

assert len(classes_aligned) == 120
print(f"Successfully aligned all 120 breed classes! Example: {classes_aligned[:5]}")

with open(os.path.join(SAVE_DIR, "classes.json"), "w", encoding="utf-8") as f:
    json.dump(classes_aligned, f, indent=2)

model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)
model.eval()

def extract_split(dataset, split_name, batch_size=128):
    print(f"\n--- Extracting {split_name} split ({len(dataset)} images, batch_size={batch_size}) ---")
    all_feats = []
    all_labels = []
    
    t0 = time.time()
    n = len(dataset)
    for i in range(0, n, batch_size):
        batch = dataset[i : min(i + batch_size, n)]
        imgs = [img.convert("RGB") for img in batch["image"]]
        labels = batch["label"]
        
        inputs = processor(images=imgs, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            feats = F.normalize(feats, dim=-1)
            
        all_feats.append(feats.cpu())
        all_labels.extend(labels)
        
        if (i // batch_size) % 10 == 0 or (i + batch_size) >= n:
            done = min(i + batch_size, n)
            elapsed = time.time() - t0
            fps = done / max(0.001, elapsed)
            print(f"[{split_name}] {done}/{n} ({done/n*100:.1f}%) | {elapsed:.1f}s | {fps:.1f} FPS")
            
    feats_tensor = torch.cat(all_feats, dim=0).float()
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    print(f"Finished {split_name}: Feats shape = {feats_tensor.shape}, Labels shape = {labels_tensor.shape}")
    return feats_tensor, labels_tensor

train_feats, train_labels = extract_split(train_ds, "train", batch_size=128)
torch.save(train_feats, os.path.join(SAVE_DIR, "train_clip.pt"))
torch.save(train_labels, os.path.join(SAVE_DIR, "train_labels.pt"))

test_feats, test_labels = extract_split(test_ds, "test", batch_size=128)
torch.save(test_feats, os.path.join(SAVE_DIR, "test_clip.pt"))
torch.save(test_labels, os.path.join(SAVE_DIR, "test_labels.pt"))

print("\n" + "="*60)
print("SUCCESS: Cached all features for Stanford Dogs Official Split!")
print(f"Directory: {SAVE_DIR}")
print(f"Train: {train_feats.shape}, Test: {test_feats.shape}")
print("="*60)

