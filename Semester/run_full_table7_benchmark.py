"""
Benchmark toàn diện tất cả các mô hình trong Table 7 qua 3 Random Seeds (1, 2, 3)
trên tập dữ liệu Oxford-IIIT Pet (3.680 trainval, 3.669 test).

Các mô hình được đánh giá:
1. CLIP Zero-Shot Baseline
2. CLIP + CoOp
3. CLIP-Adapter
4. Tip-Adapter-F
5. Gemma-CoOp (Compact Trait)
6. Gemma-CLIP Full (Adapter + Centroid Dual-Stream)
7. Supervised ResNet-50 Cosine Classifier
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
OUTPUT_JSON = "Semester/results/table7_full_benchmark_results.json"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ==============================================================================
# 1. LOAD DATA & FEATURES
# ==============================================================================
def load_all_features():
    with open(os.path.join(SAVE_DIR, "classes.json"), "r") as f:
        classes = json.load(f)
    with open(ATTR_FILE, "r", encoding="utf-8") as f:
        attributes = json.load(f)
        
    train_clip = torch.load(os.path.join(SAVE_DIR, "train_clip.pt")).float().to(device)
    test_clip = torch.load(os.path.join(SAVE_DIR, "test_clip.pt")).float().to(device)
    train_resnet = torch.load(os.path.join(SAVE_DIR, "train_resnet.pt")).float().to(device)
    test_resnet = torch.load(os.path.join(SAVE_DIR, "test_resnet.pt")).float().to(device)
    train_labels = torch.load(os.path.join(SAVE_DIR, "train_labels.pt")).to(device)
    test_labels = torch.load(os.path.join(SAVE_DIR, "test_labels.pt")).to(device)
    
    return classes, attributes, train_clip, test_clip, train_resnet, test_resnet, train_labels, test_labels

# ==============================================================================
# 2. MODEL DEFINITIONS
# ==============================================================================

# (A) CLIP-Adapter
class CLIPAdapter(nn.Module):
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

# (B) Tip-Adapter-F (Tuned Cache)
class TipAdapterF(nn.Module):
    def __init__(self, train_keys, train_labels, text_features, alpha=1.0, beta=5.5):
        super().__init__()
        self.register_buffer("text_features", text_features.detach().clone())
        self.num_classes = text_features.shape[0]
        self.alpha = alpha
        self.beta = beta
        cache_values = F.one_hot(train_labels, num_classes=self.num_classes).float()
        self.register_buffer("cache_values", cache_values)
        self.cache_keys = nn.Parameter(train_keys.clone())

    def forward(self, x):
        img_norm = F.normalize(x, dim=-1)
        keys_norm = F.normalize(self.cache_keys, dim=-1)
        sim_cache = img_norm @ keys_norm.t()
        affinity = torch.exp(-self.beta * (1.0 - sim_cache))
        logits_cache = affinity @ self.cache_values
        logits_clip = (img_norm @ self.text_features.t()) * 100.0
        return logits_clip + self.alpha * logits_cache

# (C) CoOp Classifier
class CoOpModel(nn.Module):
    def __init__(self, clip_model, num_ctx=4):
        super().__init__()
        self.clip_model = clip_model
        self.num_ctx = num_ctx
        self.ctx = nn.Parameter(torch.randn(num_ctx, 512, device=device) * 0.02)
        self.orig_forward = clip_model.text_model.embeddings.token_embedding.forward
        
        def custom_forward(input_ids, *args, **kwargs):
            embeds = self.orig_forward(input_ids, *args, **kwargs).clone()
            for i in range(self.num_ctx):
                mask = (input_ids == (49400 + i))
                if mask.any():
                    embeds[mask] = self.ctx[i].to(embeds.dtype)
            return embeds
        self.clip_model.text_model.embeddings.token_embedding.forward = custom_forward

    def get_text_features(self, input_ids):
        tf = self.clip_model.get_text_features(input_ids=input_ids)
        return F.normalize(tf, dim=-1)

    def forward(self, img_f, input_ids):
        text_f = self.get_text_features(input_ids)
        return (img_f @ text_f.t()) * self.clip_model.logit_scale.exp()

# (D) Gemma-CLIP Full (Dual-Stream Visual Adapter + Centroid Bank)
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

# (E) ResNet-50 Cosine Classifier
class ResNetCosineClassifier(nn.Module):
    def __init__(self, in_features=2048, num_classes=37, tau=10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.tau = nn.Parameter(torch.tensor(tau))

    def forward(self, x):
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.weight, dim=-1)
        return self.tau * (x_norm @ w_norm.t())

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================
def sample_stratified(feats, labels, k, num_classes=37, seed=42):
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

def make_coop_token_ids(texts, tokenizer, num_ctx=4):
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
        pad_len = 77 - len(seq)
        if pad_len > 0:
            seq = torch.cat([seq, torch.zeros(pad_len, dtype=torch.long)])
        coop_input_ids.append(seq)
    return torch.stack(coop_input_ids).to(device)

# ==============================================================================
# 4. MAIN BENCHMARK EXECUTION
# ==============================================================================
def main():
    print("=" * 80)
    print("🔬 TOÀN BỘ BENCHMARK TABLE 7: ĐÁNH GIÁ 7 MÔ HÌNH QUA 3 SEEDS (OXFORD PETS)")
    print("=" * 80)

    classes, attributes, train_clip, test_clip, train_resnet, test_resnet, train_labels, test_labels = load_all_features()
    print(f"Dataset: Oxford-IIIT Pet ({len(classes)} classes, 3.680 train, 3.669 test)")

    model_id = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    clip_model = CLIPModel.from_pretrained(model_id).to(device)
    clip_model.eval()

    # Handcrafted prompt representations
    prompts_hand = [f"a photo of a {c}, a type of pet." for c in classes]
    inp_h = tokenizer(prompts_hand, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        tf_hand = F.normalize(clip_model.get_text_features(**inp_h), dim=-1)

    # Centroid bank (5-axis Gemma attributes)
    templates = [
        "head and ears of {name}: {head_and_ears}",
        "eyes of {name}: {eyes}",
        "coat texture and markings of {name}: {coat}",
        "body structure of {name}: {body}",
        "distinguishing trait of {name}: {distinguishing_mark}"
    ]
    c_weights = []
    for c in classes:
        k = c.lower().replace(" ", "_")
        attr = attributes[k]["attributes"]
        prompts = [t.format(name=c, **attr) for t in templates]
        inp = tokenizer(prompts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        with torch.no_grad():
            tf = F.normalize(clip_model.get_text_features(**inp), dim=-1)
            c_weights.append(F.normalize(tf.mean(dim=0, keepdim=True), dim=-1))
    W_centroid = torch.cat(c_weights, dim=0)

    # CoOp & Gemma-CoOp token IDs
    texts_coop_base = [f"{c}" for c in classes]
    texts_coop_gemma = []
    for c in classes:
        k = c.lower().replace(" ", "_")
        d = attributes[k]
        texts_coop_gemma.append(f"{c}, {d.get('species', 'pet')}, recognized by {d['attributes']['distinguishing_mark']}")

    ids_coop_base = make_coop_token_ids(texts_coop_base, tokenizer, num_ctx=4)
    ids_coop_gemma = make_coop_token_ids(texts_coop_gemma, tokenizer, num_ctx=4)

    # Zero-shot baseline accuracy
    with torch.no_grad():
        zs_preds = (F.normalize(test_clip, dim=-1) @ tf_hand.t()).argmax(dim=-1)
        zs_acc = (zs_preds == test_labels).float().mean().item() * 100.0

    print(f"Baseline Handcrafted Zero-Shot Acc: {zs_acc:.2f}%\n")

    seeds = [1, 2, 3]
    shots = [1, 2, 4, 8, 16, "full"]
    models_to_run = [
        "CLIP + CoOp",
        "CLIP-Adapter",
        "Tip-Adapter-F",
        "Gemma-CoOp",
        "Gemma-CLIP Full",
        "ResNet-50 Cosine"
    ]

    all_results = {m: {} for m in models_to_run}

    print("🚀 Bắt đầu quá trình huấn luyện đa hạt nhân qua 3 Seeds...")

    for k in shots:
        shot_key = f"{k}_shot" if k != "full" else "full_trainval"
        print(f"\n--- Đang đánh giá ngân sách: {shot_key} ---")

        for m_name in models_to_run:
            seed_accs = []
            for s in seeds:
                torch.manual_seed(s)
                np.random.seed(s)

                if m_name == "CLIP + CoOp":
                    bx, by = sample_stratified(train_clip, train_labels, k, seed=s)
                    model = CoOpModel(clip_model, num_ctx=4).to(device)
                    opt = torch.optim.AdamW([model.ctx], lr=2e-3, weight_decay=1e-4)
                    crit = nn.CrossEntropyLoss()
                    for ep in range(25):
                        model.train()
                        opt.zero_grad()
                        loss = crit(model(bx, ids_coop_base), by)
                        loss.backward()
                        opt.step()
                    model.eval()
                    with torch.no_grad():
                        acc = (model(test_clip, ids_coop_base).argmax(-1) == test_labels).float().mean().item() * 100.0

                elif m_name == "CLIP-Adapter":
                    bx, by = sample_stratified(train_clip, train_labels, k, seed=s)
                    model = CLIPAdapter(tf_hand, bottleneck=64, alpha=0.2).to(device)
                    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                    crit = nn.CrossEntropyLoss()
                    for ep in range(25):
                        model.train()
                        opt.zero_grad()
                        loss = crit(model(bx), by)
                        loss.backward()
                        opt.step()
                    model.eval()
                    with torch.no_grad():
                        acc = (model(test_clip).argmax(-1) == test_labels).float().mean().item() * 100.0

                elif m_name == "Tip-Adapter-F":
                    bx, by = sample_stratified(train_clip, train_labels, k, seed=s)
                    model = TipAdapterF(bx, by, tf_hand, alpha=1.0, beta=5.5).to(device)
                    opt = torch.optim.AdamW([model.cache_keys], lr=1e-3, weight_decay=1e-4)
                    crit = nn.CrossEntropyLoss()
                    for ep in range(20):
                        model.train()
                        opt.zero_grad()
                        loss = crit(model(bx), by)
                        loss.backward()
                        opt.step()
                    model.eval()
                    with torch.no_grad():
                        acc = (model(test_clip).argmax(-1) == test_labels).float().mean().item() * 100.0

                elif m_name == "Gemma-CoOp":
                    bx, by = sample_stratified(train_clip, train_labels, k, seed=s)
                    model = CoOpModel(clip_model, num_ctx=4).to(device)
                    opt = torch.optim.AdamW([model.ctx], lr=2e-3, weight_decay=1e-4)
                    crit = nn.CrossEntropyLoss()
                    for ep in range(25):
                        model.train()
                        opt.zero_grad()
                        loss = crit(model(bx, ids_coop_gemma), by)
                        loss.backward()
                        opt.step()
                    model.eval()
                    with torch.no_grad():
                        acc = (model(test_clip, ids_coop_gemma).argmax(-1) == test_labels).float().mean().item() * 100.0

                elif m_name == "Gemma-CLIP Full":
                    bx, by = sample_stratified(train_clip, train_labels, k, seed=s)
                    model = GemmaCLIPFull(W_centroid, tf_hand, bottleneck=64, alpha=0.2, beta=0.35).to(device)
                    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                    crit = nn.CrossEntropyLoss()
                    for ep in range(30):
                        model.train()
                        opt.zero_grad()
                        loss = crit(model(bx), by)
                        loss.backward()
                        opt.step()
                    model.eval()
                    with torch.no_grad():
                        acc = (model(test_clip).argmax(-1) == test_labels).float().mean().item() * 100.0

                elif m_name == "ResNet-50 Cosine":
                    bx, by = sample_stratified(train_resnet, train_labels, k, seed=s)
                    model = ResNetCosineClassifier(in_features=2048, num_classes=37, tau=10.0).to(device)
                    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
                    crit = nn.CrossEntropyLoss()
                    for ep in range(30):
                        model.train()
                        opt.zero_grad()
                        loss = crit(model(bx), by)
                        loss.backward()
                        opt.step()
                    model.eval()
                    with torch.no_grad():
                        acc = (model(test_resnet).argmax(-1) == test_labels).float().mean().item() * 100.0

                seed_accs.append(acc)

            m = float(np.mean(seed_accs))
            s_val = float(np.std(seed_accs, ddof=1))
            all_results[m_name][shot_key] = {
                "runs": [round(a, 2) for a in seed_accs],
                "mean": round(m, 2),
                "std": round(s_val, 2)
            }
            print(f"  [{m_name:<18}] -> Runs: {[round(a,2) for a in seed_accs]} -> {m:6.2f} ± {s_val:4.2f}%")

    # ==============================================================================
    # 5. IN RA BẢNG TỔNG KẾT THEO FORMAT TABLE 7
    # ==============================================================================
    print("\n" + "=" * 95)
    print("📊 BẢNG TỔNG HỢP MULTI-SEED CHUẨN XÁC CỦA TABLE 7 (MEAN ± STD)")
    print("=" * 95)
    
    header = f"{'Budget (K)':<14} | " + " | ".join([f"{m[:15]:<15}" for m in models_to_run])
    print(header)
    print("-" * len(header))

    for k in shots:
        shot_key = f"{k}_shot" if k != "full" else "full_trainval"
        row = f"{shot_key:<14} | "
        for m_name in models_to_run:
            res = all_results[m_name][shot_key]
            row += f"{res['mean']:5.2f} ± {res['std']:4.2f}% | "
        print(row)

    print("=" * 95)

    # Lưu kết quả đầy đủ
    summary_out = {
        "dataset": "Oxford-IIIT Pet",
        "clip_handcrafted_zero_shot": round(zs_acc, 2),
        "seeds": seeds,
        "models": all_results
    }
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)

    print(f"\n✅ Đã lưu kết quả toàn bộ 108 lượt chạy vào: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()

