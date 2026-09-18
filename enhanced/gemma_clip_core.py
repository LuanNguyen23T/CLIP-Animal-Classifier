"""
enhanced/gemma_clip_core.py

Module cốt lõi phục vụ tích hợp tri thức thuộc tính Gemma 4 và CLIP ViT-B/32:
- Xây dựng Hierarchical Prompt Bank & Concept Bank
- Tính toán Centroid Text Embeddings W_c
- Hiện thực Concept Bottleneck Model (CBM)
- Hiện thực Gemma Attribute-Aware Residual Adapter
- Tính toán đầy đủ metrics (Top-1, Top-5, Precision, Recall, F1, Latency, Throughput)
"""

import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, top_k_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def generate_hierarchical_prompts(breed_data: dict) -> list:
    """
    Sinh tập hợp các prompt đa tầng bậc và chi tiết giải phẫu từ tri thức Gemma 4.
    """
    breed_raw = breed_data["breed"]
    human_name = breed_data.get("human_name", breed_raw.replace("_", " ").title())
    species = breed_data.get("species", "animal")
    attrs = breed_data.get("attributes", {})

    head = attrs.get("head_and_ears", "")
    eyes = attrs.get("eyes", "")
    coat = attrs.get("coat", "")
    body = attrs.get("body", "")
    dist = attrs.get("distinguishing_mark", "")

    prompts = [
        f"a clear photo of a {human_name}, a type of {species}.",
        f"a photo of a {species} belonging to the {human_name} breed.",
        f"a close-up photo of a {human_name} with {head}.",
        f"a photo showing the eyes and gaze of a {human_name}: {eyes}.",
        f"a photo displaying the coat texture and coloration of a {human_name}: {coat}.",
        f"a full-body photo of a {human_name} showing its stance and build: {body}.",
        f"a distinctive photo of a {human_name}, recognized by {dist}.",
        f"a photo of a {human_name} {species} with {coat} and {head}."
    ]
    return [p.strip() for p in prompts if p.strip()]


def generate_concept_prompts(breed_data: dict) -> list:
    """
    Sinh 5 concept prompt đơn lẻ phục vụ Concept Bottleneck Model (CBM).
    """
    human_name = breed_data.get("human_name", breed_data["breed"].replace("_", " ").title())
    attrs = breed_data.get("attributes", {})
    return [
        (f"head and ears of {human_name}", f"a photo focusing on the head and ears of a {human_name}: {attrs.get('head_and_ears', '')}"),
        (f"eyes of {human_name}", f"a photo focusing on the eyes of a {human_name}: {attrs.get('eyes', '')}"),
        (f"coat of {human_name}", f"a photo displaying the coat and fur pattern of a {human_name}: {attrs.get('coat', '')}"),
        (f"body build of {human_name}", f"a photo showing the body build and posture of a {human_name}: {attrs.get('body', '')}"),
        (f"distinguishing trait of {human_name}", f"a photo of the distinguishing visual feature of a {human_name}: {attrs.get('distinguishing_mark', '')}")
    ]


def build_gemma_feature_banks(attributes_dict: dict, class_names: list, clip_model, clip_processor, device: str):
    """
    Mã hóa toàn bộ prompt thuộc tính qua CLIP Text Encoder để xây dựng:
    - W_centroid: (37, 512) - Ma trận trọng tâm biểu diễn lớp kết hợp đa thuộc tính
    - W_concepts: (185, 512) - Ngân hàng vector khái niệm giải phẫu
    - concept_names: danh sách 185 tên khái niệm
    """
    clip_model.eval()
    centroids = []
    concept_vectors = []
    concept_names = []

    print("⏳ Đang mã hóa Prompt Bank & Concept Bank từ tri thức Gemma qua CLIP Text Encoder...")
    with torch.no_grad():
        for c_idx, c_name in enumerate(class_names):
            breed_key = c_name.lower().replace(" ", "_")
            b_data = attributes_dict.get(breed_key, {})
            if not b_data or "attributes" not in b_data:
                # Fallback nếu thiếu
                b_data = {
                    "breed": breed_key,
                    "human_name": c_name,
                    "species": "pet",
                    "attributes": {
                        "head_and_ears": "typical head shape",
                        "eyes": "expressive eyes",
                        "coat": "characteristic fur pattern",
                        "body": "well-proportioned body",
                        "distinguishing_mark": "distinctive breed traits"
                    }
                }

            # 1. Centroid prompts
            prompts = generate_hierarchical_prompts(b_data)
            tokens = clip_processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            text_embeds = clip_model.get_text_features(**tokens)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            centroid = text_embeds.mean(dim=0)
            centroid = centroid / centroid.norm(dim=-1, keepdim=True)
            centroids.append(centroid)

            # 2. Concept prompts (5 concepts per breed)
            concepts = generate_concept_prompts(b_data)
            c_texts = [item[1] for item in concepts]
            c_tokens = clip_processor(text=c_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            c_embeds = clip_model.get_text_features(**c_tokens)
            c_embeds = c_embeds / c_embeds.norm(dim=-1, keepdim=True)
            concept_vectors.append(c_embeds)
            for item in concepts:
                concept_names.append(f"{c_name} - {item[0]}")

    W_centroid = torch.stack(centroids, dim=0) # (37, 512)
    W_concepts = torch.cat(concept_vectors, dim=0) # (185, 512)
    print(f"✅ Hoàn tất! Shape W_centroid: {W_centroid.shape} | Shape W_concepts: {W_concepts.shape}")
    return W_centroid, W_concepts, concept_names


class ConceptBottleneckModel(nn.Module):
    """
    Concept Bottleneck Model (CBM) dựa trên 185 thuộc tính thị giác của Gemma.
    Chiếu f_img vào không gian concept s = f_img @ W_concepts.T,
    sau đó tổng hợp concept về từng lớp: logit = mean(5 concepts of class).
    """
    def __init__(self, W_concepts: torch.Tensor, num_classes: int = 37, concepts_per_class: int = 5, logit_scale: float = 100.0):
        super().__init__()
        self.register_buffer("W_concepts", W_concepts.clone()) # (185, 512)
        self.num_classes = num_classes
        self.concepts_per_class = concepts_per_class
        self.logit_scale = logit_scale

        # Tạo ma trận ánh xạ Concept -> Class (185, 37)
        pooling_matrix = torch.zeros(num_classes * concepts_per_class, num_classes)
        for c in range(num_classes):
            pooling_matrix[c * concepts_per_class : (c + 1) * concepts_per_class, c] = 1.0 / concepts_per_class
        self.register_buffer("pooling_matrix", pooling_matrix)

    def forward(self, x: torch.Tensor):
        x_norm = x / x.norm(dim=-1, keepdim=True)
        # Điểm tương đồng concept s: (B, 185)
        concept_scores = x_norm @ self.W_concepts.t()
        # Ánh xạ về logits lớp: (B, 37)
        class_logits = (concept_scores @ self.pooling_matrix) * self.logit_scale
        return class_logits, concept_scores


class GemmaAttributeAdapter(nn.Module):
    """
    Gemma Attribute-Aware Residual Adapter:
    - Nhánh Visual Residual: MLP 512 -> 128 -> 512 với LayerNorm, GELU, Dropout
    - Cơ chế thặng dư thích ứng: f_adapt = Normalize((1 - alpha) * f_img + alpha * f_res)
    - Nhánh Concept Guidance: dung hợp điểm số từ W_concepts vào logits phân loại
    """
    def __init__(
        self,
        W_centroid: torch.Tensor,
        W_concepts: torch.Tensor,
        bottleneck_dim: int = 128,
        num_classes: int = 37,
        alpha: float = 0.25,
        dropout: float = 0.1,
        logit_scale: float = 100.0
    ):
        super().__init__()
        input_dim = W_centroid.shape[1] # 512
        self.register_buffer("W_centroid", W_centroid.clone())
        self.register_buffer("W_concepts", W_concepts.clone())
        self.logit_scale = logit_scale
        self.alpha = alpha

        # Visual Adapter MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, input_dim)
        )
        # Khởi tạo trọng số tầng cuối gần 0 để ban đầu hoạt động như identity
        nn.init.constant_(self.mlp[4].weight, 0.0)
        nn.init.constant_(self.mlp[4].bias, 0.0)

        # Concept guidance head
        num_concepts = W_concepts.shape[0] # 185
        self.concept_head = nn.Linear(num_concepts, num_classes, bias=False)
        # Khởi tạo concept_head bằng ma trận pooling trung bình
        with torch.no_grad():
            pooling_init = torch.zeros(num_classes, num_concepts)
            for c in range(num_classes):
                pooling_init[c, c * 5 : (c + 1) * 5] = 1.0 / 5.0
            self.concept_head.weight.copy_(pooling_init)

        # Trọng số dung hợp giữa Text Centroid Logits và Concept Guidance Logits
        self.fusion_weight = nn.Parameter(torch.tensor(0.15))

    def forward(self, x: torch.Tensor):
        x_norm = x / x.norm(dim=-1, keepdim=True)
        res = self.mlp(x_norm)
        x_adapted = (1.0 - self.alpha) * x_norm + self.alpha * res
        x_adapted = x_adapted / x_adapted.norm(dim=-1, keepdim=True)

        # Logits từ Centroid Bank
        logits_centroid = (x_adapted @ self.W_centroid.t()) * self.logit_scale

        # Logits từ Concept Guidance Bank
        concept_scores = x_adapted @ self.W_concepts.t()
        logits_concept = self.concept_head(concept_scores) * self.logit_scale

        # Dung hợp hai dòng tri thức
        logits_final = logits_centroid + self.fusion_weight * logits_concept
        return logits_final


def evaluate_predictions(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    Tính toán bộ chỉ số hoàn chỉnh: Top-1 Acc, Top-5 Acc, Macro Precision, Recall, F1.
    """
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    preds = np.argmax(probs, axis=-1)
    y_true = targets.cpu().numpy()

    top1 = accuracy_score(y_true, preds) * 100.0
    top5 = top_k_accuracy_score(y_true, probs, k=5) * 100.0
    prec = precision_score(y_true, preds, average="macro", zero_division=0) * 100.0
    rec = recall_score(y_true, preds, average="macro", zero_division=0) * 100.0
    f1 = f1_score(y_true, preds, average="macro", zero_division=0) * 100.0

    return {
        "Top1_Acc": top1,
        "Top5_Acc": top5,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "Predictions": preds,
        "Probabilities": probs
    }


def train_gemma_adapter(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    W_centroid: torch.Tensor,
    W_concepts: torch.Tensor,
    device: str,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    label_smoothing: float = 0.05
):
    """
    Huấn luyện GemmaAttributeAdapter với Early Stopping và Cosine Annealing.
    """
    model = GemmaAttributeAdapter(
        W_centroid=W_centroid,
        W_concepts=W_concepts,
        bottleneck_dim=128,
        num_classes=37,
        alpha=0.25,
        dropout=0.1
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    dataset = TensorDataset(train_features.to(device), train_labels.to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)

        scheduler.step()
        epoch_loss = total_loss / len(train_features)
        history["train_loss"].append(epoch_loss)

        # Validation
        if val_features is not None and val_labels is not None:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_features.to(device))
                val_metrics = evaluate_predictions(val_logits, val_labels)
                v_acc = val_metrics["Top1_Acc"]
                history["val_acc"].append(v_acc)
                if v_acc > best_val_acc:
                    best_val_acc = v_acc
                    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history, best_val_acc
