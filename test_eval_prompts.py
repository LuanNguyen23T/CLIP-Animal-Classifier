import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import json
import torchvision
from transformers import CLIPModel, CLIPProcessor
import numpy as np

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# Load models and data
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

with open("enhanced/pets_attributes_gemma.json", "r") as f:
    attributes_dict = json.load(f)

pets_test = torchvision.datasets.OxfordIIITPet(root="dataset", split="test", download=False)
class_names = pets_test.classes
num_classes = len(class_names)

test_features_raw = torch.load("dataset/test_features.pt").to(device)
test_labels = torch.load("dataset/test_labels.pt").to(device)
trainval_features = torch.load("dataset/trainval_features.pt").to(device)
trainval_labels = torch.load("dataset/trainval_labels.pt").to(device)

print(f"Test features: {test_features_raw.shape}, Test labels: {test_labels.shape}")

# Stratified 9:1 split
torch.manual_seed(42)
np.random.seed(42)
train_idx, val_idx = [], []
for c in range(num_classes):
    c_indices = (trainval_labels == c).nonzero(as_tuple=True)[0]
    c_indices = c_indices[torch.randperm(len(c_indices))]
    split_pt = int(len(c_indices) * 0.9)
    train_idx.extend(c_indices[:split_pt].tolist())
    val_idx.extend(c_indices[split_pt:].tolist())

train_f_90 = trainval_features[train_idx]
train_l_90 = trainval_labels[train_idx]
val_f_10 = trainval_features[val_idx]
val_l_10 = trainval_labels[val_idx]

# Prompts
p1_texts = [f"{c}." for c in class_names]
p2_texts = [f"a photo of a {c}." for c in class_names]
p3_texts = []
p4_texts = []
for c in class_names:
    k = c.lower().replace(' ', '_')
    d = attributes_dict.get(k, {})
    sp = d.get('species', 'pet')
    dist = d.get('attributes', {}).get('distinguishing_mark', '')
    p3_texts.append(f"a photo of a {c}, a type of {sp}.")
    p4_texts.append(f"a photo of a {c}, recognized by {dist}.")

def eval_zero_shot(prompts):
    toks = clip_processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        t_f = clip_model.get_text_features(**toks)
        t_f = t_f / t_f.norm(dim=-1, keepdim=True)
        logits = (test_features_raw @ t_f.t()) * clip_model.logit_scale.exp()
        preds = logits.argmax(dim=-1)
    
    per_class_acc = {}
    for c in range(num_classes):
        mask = (test_labels == c)
        acc = (preds[mask] == test_labels[mask]).float().mean().item() * 100.0
        per_class_acc[c] = acc
    total_acc = (preds == test_labels).float().mean().item() * 100.0
    return per_class_acc, total_acc

p1_acc, p1_tot = eval_zero_shot(p1_texts)
p2_acc, p2_tot = eval_zero_shot(p2_texts)
p3_acc, p3_tot = eval_zero_shot(p3_texts)
p4_acc, p4_tot = eval_zero_shot(p4_texts)
print(f"Total zero shot accs: P1={p1_tot:.2f}%, P2={p2_tot:.2f}%, P3={p3_tot:.2f}%, P4={p4_tot:.2f}%")

# CoOp models
def make_coop_ids(texts, processor, num_ctx=4):
    coop_input_ids = []
    for text in texts:
        tokens = processor.tokenizer.encode(text, add_special_tokens=False)
        ctx_tokens = list(range(49400, 49400 + num_ctx))
        seq = [processor.tokenizer.bos_token_id] + ctx_tokens + tokens + [processor.tokenizer.eos_token_id]
        seq = torch.tensor(seq, dtype=torch.long)
        if len(seq) > 77:
            seq = torch.cat([seq[:76], torch.tensor([processor.tokenizer.eos_token_id], dtype=torch.long)])
        elif len(seq) < 77:
            padding_len = 77 - len(seq)
            seq = torch.cat([seq, torch.zeros(padding_len, dtype=torch.long)])
        coop_input_ids.append(seq)
    return torch.stack(coop_input_ids).to(device)

class CoOpClassifierModel(nn.Module):
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
    def forward(self, img_f, input_ids):
        text_f = self.clip_model.get_text_features(input_ids=input_ids)
        text_f = text_f / text_f.norm(dim=-1, keepdim=True)
        return (img_f @ text_f.t()) * self.clip_model.logit_scale.exp()

texts_coop_base = [f"{c}" for c in class_names]
texts_coop_gemma = []
for c in class_names:
    k = c.lower().replace(' ', '_')
    d = attributes_dict.get(k, {})
    texts_coop_gemma.append(f"{c}, {d.get('species', 'pet')}, recognized by {d.get('attributes', {}).get('distinguishing_mark', '')}")

ids_coop_base = make_coop_ids(texts_coop_base, clip_processor, num_ctx=4)
ids_coop_gemma = make_coop_ids(texts_coop_gemma, clip_processor, num_ctx=4)

def train_coop(coop_ids, epochs=40, lr=0.002, seed=42):
    torch.manual_seed(seed)
    model = CoOpClassifierModel(clip_model, num_ctx=4).to(device)
    optimizer = torch.optim.AdamW([model.ctx], lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(train_f_90, train_l_90), batch_size=256, shuffle=True)
    best_v = 0.0
    best_w = None
    for ep in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx, coop_ids), by)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vacc = (model(val_f_10, coop_ids).argmax(-1) == val_l_10).float().mean().item() * 100
            if vacc >= best_v:
                best_v = vacc
                best_w = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_w)
    return model

print("Training baseline CoOp...")
model_base = train_coop(ids_coop_base)
print("Training Gemma-CoOp...")
model_gemma = train_coop(ids_coop_gemma)

def eval_model(model, coop_ids):
    model.eval()
    with torch.no_grad():
        logits = model(test_features_raw, coop_ids)
        preds = logits.argmax(dim=-1)
    per_class_acc = {}
    for c in range(num_classes):
        mask = (test_labels == c)
        acc = (preds[mask] == test_labels[mask]).float().mean().item() * 100.0
        per_class_acc[c] = acc
    total_acc = (preds == test_labels).float().mean().item() * 100.0
    return per_class_acc, total_acc

coop_base_acc, coop_base_tot = eval_model(model_base, ids_coop_base)
coop_gemma_acc, coop_gemma_tot = eval_model(model_gemma, ids_coop_gemma)
print(f"CoOp Base Total: {coop_base_tot:.2f}%, Gemma-CoOp Total: {coop_gemma_tot:.2f}%")

# Save results to json
all_res = {
    "classes": class_names,
    "p1_acc": p1_acc,
    "p2_acc": p2_acc,
    "p3_acc": p3_acc,
    "p4_acc": p4_acc,
    "coop_base_acc": coop_base_acc,
    "coop_gemma_acc": coop_gemma_acc,
    "totals": {
        "p1": p1_tot,
        "p2": p2_tot,
        "p3": p3_tot,
        "p4": p4_tot,
        "coop_base": coop_base_tot,
        "coop_gemma": coop_gemma_tot
    }
}
with open("test_eval_results.json", "w") as f:
    json.dump(all_res, f, indent=2)
print("Saved to test_eval_results.json")
