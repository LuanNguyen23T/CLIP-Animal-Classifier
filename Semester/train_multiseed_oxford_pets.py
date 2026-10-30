"""
Script huấn luyện và đánh giá Gemma-CLIP qua nhiều Random Seeds (Seeds 1, 2, 3)
trên tập dữ liệu Oxford-IIIT Pet (3.680 trainval, 3.669 test).
Tự động lưu log chi tiết từng seed, tính Mean ± Std để bảo vệ trước phản biện.
"""

import os
import json
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

SAVE_DIR = "Semester/cached_features/oxford_pets"
ATTR_FILE = "Semester/attributes/oxford_pets_attributes.json"
OUTPUT_LOG_FILE = "Semester/results/oxford_pets_multiseed_logs.json"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_data():
    with open(os.path.join(SAVE_DIR, "classes.json"), "r") as f:
        classes = json.load(f)
    train_feats = torch.load(os.path.join(SAVE_DIR, "train_clip.pt")).float()
    train_labels = torch.load(os.path.join(SAVE_DIR, "train_labels.pt"))
    test_feats = torch.load(os.path.join(SAVE_DIR, "test_clip.pt")).float()
    test_labels = torch.load(os.path.join(SAVE_DIR, "test_labels.pt"))
    
    with open(ATTR_FILE, "r", encoding="utf-8") as f:
        attributes = json.load(f)
        
    return classes, train_feats, train_labels, test_feats, test_labels, attributes

def build_centroid_and_handcrafted_features(classes, attributes, clip_model, tokenizer):
    print("🧠 Đang xây dựng Centroid Bank và Handcrafted text representations...")
    # 1. Handcrafted baseline prompts
    prompts_hand = [f"a photo of a {c}, a type of pet." for c in classes]
    inp_h = tokenizer(prompts_hand, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        tf_hand = F.normalize(clip_model.get_text_features(**inp_h), dim=-1)

    # 2. Gemma 5-Axis Morphological Centroid Bank
    templates = [
        "head and ears of {name}: {head_and_ears}",
        "eyes of {name}: {eyes}",
        "coat texture and markings of {name}: {coat}",
        "body structure of {name}: {body}",
        "distinguishing trait of {name}: {distinguishing_mark}"
    ]

    centroid_weights = []
    for c in classes:
        key = c.lower().replace(" ", "_")
        attr = attributes[key]["attributes"]
        prompts = [t.format(name=c, **attr) for t in templates]
        inp = tokenizer(prompts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        with torch.no_grad():
            tf = F.normalize(clip_model.get_text_features(**inp), dim=-1)
            c_mean = F.normalize(tf.mean(dim=0, keepdim=True), dim=-1)
            centroid_weights.append(c_mean)

    centroid_weights = torch.cat(centroid_weights, dim=0)
    return tf_hand, centroid_weights

def sample_k_shot_stratified(feats, labels, k, num_classes=37, seed=42):
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

class GemmaResidualAdapter(nn.Module):
    def __init__(self, target_text_features, bottleneck=64, alpha=0.2):
        super().__init__()
        self.register_buffer("text_features", target_text_features.detach().clone())
        self.alpha = alpha
        self.mlp = nn.Sequential(
            nn.Linear(512, bottleneck, bias=False),
            nn.ReLU(),
            nn.Linear(bottleneck, 512, bias=False)
        )
        nn.init.constant_(self.mlp[2].weight, 0.0)

    def forward(self, x):
        x_norm = F.normalize(x, dim=-1)
        res = self.mlp(x_norm)
        adapted = F.normalize((1.0 - self.alpha) * x_norm + self.alpha * res, dim=-1)
        return (adapted @ self.text_features.t()) * 100.0

def train_one_run(train_x, train_y, test_x, test_y, text_weights, epochs=30, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    model = GemmaResidualAdapter(text_weights, bottleneck=64, alpha=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    batch_size = 128 if len(train_x) > 128 else len(train_x)
    n = len(train_x)

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            bx, by = train_x[idx], train_y[idx]
            logits = model(bx)
            loss = criterion(logits, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(test_x).argmax(dim=-1)
        acc = (preds == test_y).float().mean().item() * 100.0
    return acc

def main():
    print("=" * 75)
    print("🐾 HUẤN LUYỆN & ĐỐI SOÁT MULTI-SEED LOGS TRÊN OXFORD-IIIT PET (3 SEEDS)")
    print("=" * 75)

    classes, train_feats, train_labels, test_feats, test_labels, attributes = load_data()
    print(f"Dataset: Oxford-IIIT Pet | Classes: {len(classes)} | Train: {len(train_feats)} | Test: {len(test_feats)}")
    
    train_feats = train_feats.to(device)
    train_labels = train_labels.to(device)
    test_feats = test_feats.to(device)
    test_labels = test_labels.to(device)

    model_id = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    clip_model = CLIPModel.from_pretrained(model_id).to(device)
    clip_model.eval()

    tf_hand, centroid_weights = build_centroid_and_handcrafted_features(
        classes, attributes, clip_model, tokenizer
    )

    # Blend handcrafted and centroid for effective prompt grounding
    target_weights = F.normalize(tf_hand + 0.35 * centroid_weights, dim=-1)

    # Evaluate zero-shot baseline
    with torch.no_grad():
        zs_preds = (F.normalize(test_feats, dim=-1) @ target_weights.t()).argmax(dim=-1)
        zs_acc = (zs_preds == test_labels).float().mean().item() * 100.0
    print(f"\n🎯 Zero-Shot Baseline (Handcrafted + Gemma Centroid Blend): {zs_acc:.2f}%\n")

    seeds = [1, 2, 3]
    shots = [1, 2, 4, 8, 16, "full"]
    results_summary = {
        "dataset": "Oxford-IIIT Pet",
        "total_classes": 37,
        "test_samples": len(test_feats),
        "seeds": seeds,
        "zero_shot_accuracy": round(zs_acc, 2),
        "few_shot_scaling": {}
    }

    print(f"{'Shot (K)':<12} | {'Seed 1 (%)':<11} | {'Seed 2 (%)':<11} | {'Seed 3 (%)':<11} | {'Mean ± Std (%)':<16}")
    print("-" * 75)

    for k in shots:
        shot_label = f"K = {k}" if k != "full" else "Full Train"
        shot_key = f"{k}_shot" if k != "full" else "full_trainval"
        accuracies = []

        for s in seeds:
            # 1. Bốc K mẫu theo seed
            supp_x, supp_y = sample_k_shot_stratified(train_feats, train_labels, k=k, seed=s)
            
            # 2. Huấn luyện adapter với seed đó
            acc = train_one_run(supp_x, supp_y, test_feats, test_labels, target_weights, epochs=30, seed=s)
            accuracies.append(acc)

        mean_acc = float(np.mean(accuracies))
        std_acc = float(np.std(accuracies, ddof=1))

        results_summary["few_shot_scaling"][shot_key] = {
            "seed_runs": {f"seed_{s}": round(acc, 2) for s, acc in zip(seeds, accuracies)},
            "mean": round(mean_acc, 2),
            "std": round(std_acc, 2)
        }

        print(f"{shot_label:<12} | {accuracies[0]:<11.2f} | {accuracies[1]:<11.2f} | {accuracies[2]:<11.2f} | {mean_acc:6.2f} ± {std_acc:4.2f}%")

    print("-" * 75)

    # Lưu kết quả ra file JSON
    os.makedirs(os.path.dirname(OUTPUT_LOG_FILE), exist_ok=True)
    with open(OUTPUT_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n✅ ĐÃ HOÀN THÀNH HUẤN LUYỆN 3 SEEDS!")
    print(f"📁 Log chi tiết đã được lưu thành công tại: {OUTPUT_LOG_FILE}")

if __name__ == "__main__":
    main()

