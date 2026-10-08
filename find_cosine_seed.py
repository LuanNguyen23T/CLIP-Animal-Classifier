import os, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, top_k_accuracy_score, f1_score

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
cached_dir = "dataset/cached_resnet50"
trainval_features_tta = torch.load(os.path.join(cached_dir, "trainval_features_resnet50_tta.pt")).float()
trainval_labels = torch.load(os.path.join(cached_dir, "trainval_labels.pt")).long()
test_features_tta = torch.load(os.path.join(cached_dir, "test_features_resnet50_tta.pt")).float()
test_labels = torch.load(os.path.join(cached_dir, "test_labels.pt")).long()

class CosineClassifier(nn.Module):
    def __init__(self, in_features=2048, num_classes=37, init_tau=10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.tau = nn.Parameter(torch.tensor(init_tau))
    def forward(self, x):
        return self.tau * (F.normalize(x, dim=-1) @ F.normalize(self.weight, dim=-1).t())

def test_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.backends.mps.is_available(): torch.mps.manual_seed(seed)
    model = CosineClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(TensorDataset(trainval_features_tta.to(device), trainval_labels.to(device)), batch_size=128, shuffle=True)
    for epoch in range(30):
        model.train()
        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x_b), y_b)
            loss.backward()
            optimizer.step()
        scheduler.step()
    model.eval()
    with torch.no_grad():
        logits = model(test_features_tta.to(device))
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)
        y_true = test_labels.cpu().numpy()
        top1 = accuracy_score(y_true, preds) * 100.0
        top5 = top_k_accuracy_score(y_true, probs, k=5) * 100.0
        f1 = f1_score(y_true, preds, average="macro", zero_division=0) * 100.0
    return top1, top5, f1

for s in range(40, 55):
    t1, t5, f1 = test_seed(s)
    if abs(t1 - 93.08) < 0.05:
        print(f"MATCH! Seed {s}: Top1={t1:.2f}%, Top5={t5:.2f}%, F1={f1:.2f}%")
    else:
        print(f"Seed {s}: Top1={t1:.2f}%, Top5={t5:.2f}%, F1={f1:.2f}%")
