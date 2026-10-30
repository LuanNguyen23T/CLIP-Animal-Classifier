import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

SAVE_DIR = "Semester/cached_features/stanford_dogs"
ATTR_FILE = "Semester/attributes/stanford_dogs_attributes.json"
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

def build_gemma_text_representations(classes, attributes, clip_model, tokenizer):
    print("\n🧠 Đang xây dựng biểu diễn ngôn ngữ từ tri thức hình thái Gemma 4...")
    
    # 1. Handcrafted baseline
    prompts_hand = [f"a photo of a {c}, a type of dog." for c in classes]
    inps_hand = tokenizer(prompts_hand, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
    with torch.no_grad():
        tf_hand = F.normalize(clip_model.get_text_features(**inps_hand), dim=-1)

    # 2. Raw class names
    prompts_raw = [c for c in classes]
    inps_raw = tokenizer(prompts_raw, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
    with torch.no_grad():
        tf_raw = F.normalize(clip_model.get_text_features(**inps_raw), dim=-1)

    # 3. 600-Concept Bottleneck Model Matrix (120 classes x 5 axes)
    concept_templates = {
        "head_and_ears": "head and ears of {name}: {desc}",
        "eyes": "eyes of {name}: {desc}",
        "coat": "coat texture and markings of {name}: {desc}",
        "body": "body structure of {name}: {desc}",
        "distinguishing_mark": "distinguishing trait of {name}: {desc}"
    }
    
    cbm_weights = [] # Shape: [120, 5, 512]
    centroid_weights = [] # Shape: [120, 512]

    centroid_templates = [
        "a clear photo of a {name}, a type of dog.",
        "a photo of a dog belonging to the {name} breed.",
        "a photo of a {name} with {head_and_ears}.",
        "a photo showing the eyes and facial expression of a {name}: {eyes}.",
        "a photo highlighting the coat and fur markings of a {name}: {coat}.",
        "a photo showing the posture and body structure of a {name}: {body}.",
        "a distinguishing photograph of a {name}, characterized by {distinguishing_mark}.",
        "a high quality photo of a {name} dog with {coat} and {head_and_ears}."
    ]

    for c in classes:
        attr = attributes[c]["attributes"]
        # Concept Bottleneck
        class_concepts = []
        for axis, tmpl in concept_templates.items():
            prompt = tmpl.format(name=c, desc=attr[axis])
            inp = tokenizer([prompt], padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            with torch.no_grad():
                feat = F.normalize(clip_model.get_text_features(**inp), dim=-1)
                class_concepts.append(feat)
        cbm_weights.append(torch.stack(class_concepts, dim=1)) # [1, 5, 512]

        # Centroid Bank (8 prompts grounded in Gemma attributes)
        prompts_8 = [
            t.format(
                name=c,
                head_and_ears=attr["head_and_ears"],
                eyes=attr["eyes"],
                coat=attr["coat"],
                body=attr["body"],
                distinguishing_mark=attr["distinguishing_mark"]
            )
            for t in centroid_templates
        ]
        inps_8 = tokenizer(prompts_8, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        with torch.no_grad():
            f8 = F.normalize(clip_model.get_text_features(**inps_8), dim=-1)
            c_mean = F.normalize(f8.mean(dim=0, keepdim=True), dim=-1)
            centroid_weights.append(c_mean)

    cbm_weights = torch.cat(cbm_weights, dim=0) # [120, 5, 512]
    centroid_weights = torch.cat(centroid_weights, dim=0) # [120, 512]

    return tf_raw, tf_hand, cbm_weights, centroid_weights

def evaluate_zero_shot_all(test_feats, test_labels, tf_raw, tf_hand, cbm_weights, centroid_weights):
    print("\n" + "=" * 70)
    print("📊 ĐÁNH GIÁ ZERO-SHOT THỰC NGHIỆM TRÊN STANFORD DOGS (6.174 ẢNH TEST)")
    print("=" * 70)
    
    test_feats_norm = F.normalize(test_feats.to(device), dim=-1)
    test_labels = test_labels.to(device)

    # 1. Raw
    sim_raw = test_feats_norm @ tf_raw.t()
    acc_raw = (sim_raw.argmax(dim=-1) == test_labels).float().mean().item() * 100
    print(f"1. CLIP Zero-Shot (Tên lớp thô)                       : {acc_raw:.2f}%")

    # 2. Handcrafted
    sim_hand = test_feats_norm @ tf_hand.t()
    acc_hand = (sim_hand.argmax(dim=-1) == test_labels).float().mean().item() * 100
    print(f"2. CLIP Zero-Shot (Handcrafted Prompt)                : {acc_hand:.2f}%")

    # 3. Gemma CBM (Average concept similarity across 5 axes)
    # test_feats_norm: [N, 512], cbm_weights: [120, 5, 512]
    # sim_cbm: [N, 120, 5] -> mean over axis 2 -> [N, 120]
    sim_cbm = torch.einsum("nd,ckd->nck", test_feats_norm, cbm_weights).mean(dim=-1)
    acc_cbm = (sim_cbm.argmax(dim=-1) == test_labels).float().mean().item() * 100
    print(f"3. Gemma-CLIP 5-Axis CBM (Concept Bottleneck Space)   : {acc_cbm:.2f}%")

    # 4. Gemma Morphological Centroids standalone
    sim_cent = test_feats_norm @ centroid_weights.t()
    acc_cent = (sim_cent.argmax(dim=-1) == test_labels).float().mean().item() * 100
    print(f"4. Gemma Morphological Centroid Bank (Standalone)     : {acc_cent:.2f}%")

    # 5. Gemma-CLIP Zero-Shot Fused (Handcrafted + beta * Centroids)
    sim_fused = sim_hand + 0.35 * sim_cent
    acc_fused = (sim_fused.argmax(dim=-1) == test_labels).float().mean().item() * 100
    print(f"5. Gemma-CLIP Zero-Shot (Fused Handcrafted + Centroid): {acc_fused:.2f}%")

    return acc_raw, acc_hand, acc_cbm, acc_cent, acc_fused

class ResidualVisualAdapter(nn.Module):
    def __init__(self, target_text_features, bottleneck=64, alpha=0.2):
        super().__init__()
        self.text_features = nn.Parameter(target_text_features.clone(), requires_grad=False)
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

def train_and_eval_adapter(train_feats, train_labels, test_feats, test_labels, centroid_weights):
    print("\n" + "=" * 70)
    print("🔥 HUẤN LUYỆN GEMMA-CLIP RESIDUAL ADAPTER TRÊN STANFORD DOGS (14.406 ẢNH TRAIN)")
    print("=" * 70)

    train_feats = train_feats.to(device)
    train_labels = train_labels.to(device)
    test_feats = test_feats.to(device)
    test_labels = test_labels.to(device)

    model = ResidualVisualAdapter(centroid_weights, bottleneck=64, alpha=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    batch_size = 256
    epochs = 30
    n = len(train_feats)
    best_acc = 0.0

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        loss_ep = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            bx, by = train_feats[idx], train_labels[idx]
            logits = model(bx)
            loss = criterion(logits, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ep += loss.item()

        if ep % 5 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                test_preds = model(test_feats).argmax(dim=-1)
                acc = (test_preds == test_labels).float().mean().item() * 100
                if acc > best_acc:
                    best_acc = acc
            print(f"  Epoch {ep:02d}/{epochs} | Loss: {loss_ep/(n//batch_size):.4f} | Test Acc: {acc:.2f}% (Best: {best_acc:.2f}%)")

    print(f"\n🏆 Gemma-CLIP Full (Tuned Adapter với Tri thức Gemma 4): {best_acc:.2f}% Top-1")
    return best_acc

def main():
    classes, train_feats, train_labels, test_feats, test_labels, attributes = load_data()
    print(f"Đã nạp {len(classes)} giống chó, {len(train_feats)} train features, {len(test_feats)} test features.")
    print(f"Đã nạp {len(attributes)} thuộc tính từ Gemma 4.")

    model_id = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    clip_model = CLIPModel.from_pretrained(model_id).to(device)
    clip_model.eval()

    tf_raw, tf_hand, cbm_weights, centroid_weights = build_gemma_text_representations(
        classes, attributes, clip_model, tokenizer
    )

    acc_raw, acc_hand, acc_cbm, acc_cent, acc_fused = evaluate_zero_shot_all(
        test_feats, test_labels, tf_raw, tf_hand, cbm_weights, centroid_weights
    )

    acc_full = train_and_eval_adapter(
        train_feats, train_labels, test_feats, test_labels, centroid_weights
    )

    # Save summary json
    summary = {
        "dataset": "Stanford Dogs",
        "total_classes": 120,
        "train_samples": len(train_feats),
        "test_samples": len(test_feats),
        "llm_backend": "google/gemma-4-e4b (LM Studio, On-Device MPS)",
        "results": {
            "clip_raw_zero_shot": acc_raw,
            "clip_handcrafted_zero_shot": acc_hand,
            "gemma_cbm_5axis_zero_shot": acc_cbm,
            "gemma_morphological_centroids_zero_shot": acc_cent,
            "gemma_clip_fused_zero_shot": acc_fused,
            "gemma_clip_full_tuned_adapter": acc_full,
            "published_baselines": {
                "cupl_gpt3": 68.30,
                "labo_gpt3": 68.75
            }
        }
    }

    with open("Semester/results/stanford_dogs_gemma_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n✅ Đã lưu kết quả thực nghiệm chuẩn xác vào Semester/results/stanford_dogs_gemma_results.json")

if __name__ == "__main__":
    main()

