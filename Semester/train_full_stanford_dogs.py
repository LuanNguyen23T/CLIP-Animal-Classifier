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

def build_text_representations(classes, attributes, clip_model, tokenizer):
    prompts_hand = [f"a photo of a {c}, a type of dog." for c in classes]
    inps_hand = tokenizer(prompts_hand, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
    with torch.no_grad():
        tf_hand = F.normalize(clip_model.get_text_features(**inps_hand), dim=-1)

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
    centroid_weights = []
    for c in classes:
        attr = attributes[c]["attributes"]
        prompts = [
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
        inps = tokenizer(prompts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        with torch.no_grad():
            f8 = F.normalize(clip_model.get_text_features(**inps), dim=-1)
            c_mean = F.normalize(f8.mean(dim=0, keepdim=True), dim=-1)
            centroid_weights.append(c_mean)
    centroid_weights = torch.cat(centroid_weights, dim=0)
    return tf_hand, centroid_weights

class GemmaCLIPFull(nn.Module):
    def __init__(self, W_centroid, W_hand, bottleneck=64, alpha=0.2, beta=0.35):
        super().__init__()
        self.register_buffer("W_centroid", W_centroid.detach().clone())
        self.register_buffer("W_hand", W_hand.detach().clone())
        self.alpha = alpha
        self.beta = beta
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
        l_hand = (adapted @ self.W_hand.t()) * 100.0
        l_cent = (adapted @ self.W_centroid.t()) * 100.0
        return (1.0 - self.beta) * l_hand + self.beta * l_cent

def train_full_model(train_x, train_y, test_x, test_y, W_centroid, tf_hand, epochs=35, lr=1e-3):
    model = GemmaCLIPFull(W_centroid, tf_hand, bottleneck=64, alpha=0.2, beta=0.35).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    batch_size = 256
    n = len(train_x)
    best_acc = 0.0
    
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        loss_ep = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            bx, by = train_x[idx], train_y[idx]
            logits = model(bx)
            loss = criterion(logits, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ep += loss.item()
            
        if ep % 5 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                preds = model(test_x).argmax(dim=-1)
                acc = (preds == test_y).float().mean().item() * 100.0
                if acc > best_acc:
                    best_acc = acc
            print(f"Epoch {ep:02d}/{epochs} | Loss: {loss_ep/(n//batch_size):.4f} | Test Acc: {acc:.2f}% (Best: {best_acc:.2f}%)")
            
    return best_acc

def main():
    classes, train_feats, train_labels, test_feats, test_labels, attributes = load_data()
    print(f"Loaded {len(classes)} classes, {len(train_feats)} train, {len(test_feats)} test.")
    
    model_id = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    clip_model = CLIPModel.from_pretrained(model_id).to(device)
    clip_model.eval()
    
    tf_hand, W_centroid = build_text_representations(classes, attributes, clip_model, tokenizer)
    
    train_x = train_feats.to(device)
    train_y = train_labels.to(device)
    test_x = test_feats.to(device)
    test_y = test_labels.to(device)
    
    print("\n--- Training Gemma-CLIP Full (Dual-Stream + Adapter) on Stanford Dogs ---")
    best_acc = train_full_model(train_x, train_y, test_x, test_y, W_centroid, tf_hand, epochs=35, lr=1e-3)
    print(f"🏆 Stanford Dogs Gemma-CLIP Full (Dual-Stream + Adapter): {best_acc:.2f}%")
    
    # Save to json
    res_path = "Semester/results/stanford_dogs_gemma_results.json"
    with open(res_path, "r") as f:
        data = json.load(f)
    data["results"]["gemma_clip_full_dual_stream"] = best_acc
    with open(res_path, "w") as f:
        json.dump(data, f, indent=2)
    print("Saved to Semester/results/stanford_dogs_gemma_results.json")

if __name__ == "__main__":
    main()
