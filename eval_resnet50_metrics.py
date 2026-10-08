import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, top_k_accuracy_score, f1_score

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

set_seed(42)

# Load cached features
cached_dir = "dataset/cached_resnet50"
trainval_features_raw = torch.load(os.path.join(cached_dir, "trainval_features_resnet50_raw.pt")).float()
trainval_features_tta = torch.load(os.path.join(cached_dir, "trainval_features_resnet50_tta.pt")).float()
trainval_labels = torch.load(os.path.join(cached_dir, "trainval_labels.pt")).long()

test_features_raw = torch.load(os.path.join(cached_dir, "test_features_resnet50_raw.pt")).float()
test_features_tta = torch.load(os.path.join(cached_dir, "test_features_resnet50_tta.pt")).float()
test_labels = torch.load(os.path.join(cached_dir, "test_labels.pt")).long()

num_classes = 37

def get_stratified_split(features, labels, train_ratio=0.9, seed=42):
    set_seed(seed)
    train_indices = []
    val_indices = []
    for c in range(num_classes):
        c_idx = (labels == c).nonzero(as_tuple=True)[0]
        n_class = len(c_idx)
        perm = torch.randperm(n_class)
        n_train = max(1, int(n_class * train_ratio))
        train_indices.append(c_idx[perm[:n_train]])
        val_indices.append(c_idx[perm[n_train:]])
    train_idx = torch.cat(train_indices)
    val_idx = torch.cat(val_indices)
    return features[train_idx], labels[train_idx], features[val_idx], labels[val_idx]

X_tr_tta_90, y_tr_90, X_val_tta_90, y_val_90 = get_stratified_split(trainval_features_tta, trainval_labels, train_ratio=0.9)
X_tr_raw_90, _, X_val_raw_90, _ = get_stratified_split(trainval_features_raw, trainval_labels, train_ratio=0.9)

def compute_metrics(logits, targets):
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    preds = np.argmax(probs, axis=-1)
    y_true = targets.cpu().numpy()
    
    top1 = accuracy_score(y_true, preds) * 100.0
    top5 = top_k_accuracy_score(y_true, probs, k=5) * 100.0
    f1 = f1_score(y_true, preds, average="macro", zero_division=0) * 100.0
    return {"top1": top1, "top5": top5, "macro_f1": f1}

# 1. Prototype Classifier
def evaluate_prototype(train_feats, train_lbls, test_feats, test_lbls, val_feats=None, val_lbls=None):
    train_norm = train_feats / train_feats.norm(dim=-1, keepdim=True)
    test_norm = test_feats / test_feats.norm(dim=-1, keepdim=True)
    
    prototypes = []
    for c in range(num_classes):
        mask = (train_lbls == c)
        proto = train_norm[mask].mean(dim=0)
        proto = proto / proto.norm()
        prototypes.append(proto)
    prototypes = torch.stack(prototypes).to(device)
    
    val_acc = None
    if val_feats is not None and val_lbls is not None:
        val_norm = (val_feats / val_feats.norm(dim=-1, keepdim=True)).to(device)
        sim_val = val_norm @ prototypes.t()
        preds_val = sim_val.argmax(dim=-1)
        val_acc = (preds_val == val_lbls.to(device)).float().mean().item() * 100
        
    sim_test = test_norm.to(device) @ prototypes.t()
    metrics = compute_metrics(sim_test, test_lbls)
    return val_acc, metrics

print("\n--- 1. PROTOTYPE ---")
_, m_proto_raw = evaluate_prototype(trainval_features_raw, trainval_labels, test_features_raw, test_labels)
print(f"Proto Raw: Top-1={m_proto_raw['top1']:.2f}%, Top-5={m_proto_raw['top5']:.2f}%, F1={m_proto_raw['macro_f1']:.2f}%")

v_p90, m_proto_tta90 = evaluate_prototype(X_tr_tta_90, y_tr_90, test_features_tta, test_labels, X_val_tta_90, y_val_90)
print(f"Proto TTA (9:1): Top-1={m_proto_tta90['top1']:.2f}%, Top-5={m_proto_tta90['top5']:.2f}%, F1={m_proto_tta90['macro_f1']:.2f}% (Val={v_p90:.2f}%)")

_, m_proto_tta_full = evaluate_prototype(trainval_features_tta, trainval_labels, test_features_tta, test_labels)
print(f"Proto TTA (Full): Top-1={m_proto_tta_full['top1']:.2f}%, Top-5={m_proto_tta_full['top5']:.2f}%, F1={m_proto_tta_full['macro_f1']:.2f}%")

# 2. Linear Probe
def train_linear_probe(train_feats, train_lbls, test_feats, test_lbls, val_feats=None, val_lbls=None, epochs=30, lr=1e-3):
    classifier = nn.Linear(2048, num_classes).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(TensorDataset(train_feats.to(device), train_lbls.to(device)), batch_size=128, shuffle=True)
    best_val_acc = 0.0
    best_weights = None
    
    for epoch in range(epochs):
        classifier.train()
        total_loss = 0.0
        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            logits = classifier(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        
        if val_feats is not None and val_lbls is not None:
            classifier.eval()
            with torch.no_grad():
                val_preds = classifier(val_feats.to(device)).argmax(dim=-1)
                val_acc = (val_preds == val_lbls.to(device)).float().mean().item() * 100
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_weights = {k: v.clone() for k, v in classifier.state_dict().items()}
                    
    if best_weights is not None:
        classifier.load_state_dict(best_weights)
        
    classifier.eval()
    with torch.no_grad():
        test_logits = classifier(test_feats.to(device))
        metrics = compute_metrics(test_logits, test_lbls)
        
    return best_val_acc if val_feats is not None else None, metrics

print("\n--- 2. LINEAR PROBE ---")
_, m_lp_raw = train_linear_probe(trainval_features_raw, trainval_labels, test_features_raw, test_labels)
print(f"LP Raw: Top-1={m_lp_raw['top1']:.2f}%, Top-5={m_lp_raw['top5']:.2f}%, F1={m_lp_raw['macro_f1']:.2f}%")

v_lp90, m_lp_tta90 = train_linear_probe(X_tr_tta_90, y_tr_90, test_features_tta, test_labels, X_val_tta_90, y_val_90)
print(f"LP TTA (9:1): Top-1={m_lp_tta90['top1']:.2f}%, Top-5={m_lp_tta90['top5']:.2f}%, F1={m_lp_tta90['macro_f1']:.2f}% (Val={v_lp90:.2f}%)")

_, m_lp_tta_full = train_linear_probe(trainval_features_tta, trainval_labels, test_features_tta, test_labels)
print(f"LP TTA (Full): Top-1={m_lp_tta_full['top1']:.2f}%, Top-5={m_lp_tta_full['top5']:.2f}%, F1={m_lp_tta_full['macro_f1']:.2f}%")

# 3. Residual Adapter
class ResNetAdapter(nn.Module):
    def __init__(self, in_features=2048, hidden_dim=512, num_classes=37, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.adapter = nn.Sequential(
            nn.Linear(in_features, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_features, bias=False),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        f_res = self.adapter(x)
        f_mixed = self.alpha * f_res + (1.0 - self.alpha) * x
        return self.fc(f_mixed)

def train_resnet_adapter(train_feats, train_lbls, test_feats, test_lbls, val_feats=None, val_lbls=None, epochs=30, lr=1e-3):
    model = ResNetAdapter(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(TensorDataset(train_feats.to(device), train_lbls.to(device)), batch_size=128, shuffle=True)
    best_val_acc = 0.0
    best_weights = None
    
    for epoch in range(epochs):
        model.train()
        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if val_feats is not None and val_lbls is not None:
            model.eval()
            with torch.no_grad():
                val_preds = model(val_feats.to(device)).argmax(dim=-1)
                val_acc = (val_preds == val_lbls.to(device)).float().mean().item() * 100
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_weights = {k: v.clone() for k, v in model.state_dict().items()}
                    
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    model.eval()
    with torch.no_grad():
        test_logits = model(test_feats.to(device))
        metrics = compute_metrics(test_logits, test_lbls)
        
    return best_val_acc if val_feats is not None else None, metrics

print("\n--- 3. RESIDUAL ADAPTER ---")
_, m_adp_raw = train_resnet_adapter(trainval_features_raw, trainval_labels, test_features_raw, test_labels)
print(f"Adapter Raw: Top-1={m_adp_raw['top1']:.2f}%, Top-5={m_adp_raw['top5']:.2f}%, F1={m_adp_raw['macro_f1']:.2f}%")

v_adp90, m_adp_tta90 = train_resnet_adapter(X_tr_tta_90, y_tr_90, test_features_tta, test_labels, X_val_tta_90, y_val_90)
print(f"Adapter TTA (9:1): Top-1={m_adp_tta90['top1']:.2f}%, Top-5={m_adp_tta90['top5']:.2f}%, F1={m_adp_tta90['macro_f1']:.2f}% (Val={v_adp90:.2f}%)")

_, m_adp_tta_full = train_resnet_adapter(trainval_features_tta, trainval_labels, test_features_tta, test_labels)
print(f"Adapter TTA (Full): Top-1={m_adp_tta_full['top1']:.2f}%, Top-5={m_adp_tta_full['top5']:.2f}%, F1={m_adp_tta_full['macro_f1']:.2f}%")

# 4. Cosine Classifier Head
class CosineClassifier(nn.Module):
    def __init__(self, in_features=2048, num_classes=37, init_tau=10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.tau = nn.Parameter(torch.tensor(init_tau))
        
    def forward(self, x):
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.weight, dim=-1)
        return self.tau * (x_norm @ w_norm.t())

def train_cosine_classifier(train_feats, train_lbls, test_feats, test_lbls, val_feats=None, val_lbls=None, epochs=30, lr=5e-3):
    model = CosineClassifier(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(TensorDataset(train_feats.to(device), train_lbls.to(device)), batch_size=128, shuffle=True)
    best_val_acc = 0.0
    best_weights = None
    
    for epoch in range(epochs):
        model.train()
        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if val_feats is not None and val_lbls is not None:
            model.eval()
            with torch.no_grad():
                val_preds = model(val_feats.to(device)).argmax(dim=-1)
                val_acc = (val_preds == val_lbls.to(device)).float().mean().item() * 100
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_weights = {k: v.clone() for k, v in model.state_dict().items()}
                    
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    model.eval()
    with torch.no_grad():
        test_logits = model(test_feats.to(device))
        metrics = compute_metrics(test_logits, test_lbls)
        
    return best_val_acc if val_feats is not None else None, metrics

print("\n--- 4. COSINE HEAD CLASSIFIER ---")
_, m_cos_raw = train_cosine_classifier(trainval_features_raw, trainval_labels, test_features_raw, test_labels)
print(f"Cosine Raw: Top-1={m_cos_raw['top1']:.2f}%, Top-5={m_cos_raw['top5']:.2f}%, F1={m_cos_raw['macro_f1']:.2f}%")

v_cos90, m_cos_tta90 = train_cosine_classifier(X_tr_tta_90, y_tr_90, test_features_tta, test_labels, X_val_tta_90, y_val_90)
print(f"Cosine TTA (9:1): Top-1={m_cos_tta90['top1']:.2f}%, Top-5={m_cos_tta90['top5']:.2f}%, F1={m_cos_tta90['macro_f1']:.2f}% (Val={v_cos90:.2f}%)")

_, m_cos_tta_full = train_cosine_classifier(trainval_features_tta, trainval_labels, test_features_tta, test_labels)
print(f"Cosine TTA (Full): Top-1={m_cos_tta_full['top1']:.2f}%, Top-5={m_cos_tta_full['top5']:.2f}%, F1={m_cos_tta_full['macro_f1']:.2f}%")
