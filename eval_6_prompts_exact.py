import torch
import json
import torchvision
from transformers import CLIPModel, CLIPProcessor

device = "mps" if torch.backends.mps.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

pets_test = torchvision.datasets.OxfordIIITPet(root="dataset", split="test", download=False)
class_names = pets_test.classes
num_classes = len(class_names)

test_features_raw = torch.load("dataset/test_features.pt").to(device)
test_labels = torch.load("dataset/test_labels.pt").to(device)

with open("enhanced/pets_attributes_gemma.json", "r") as f:
    attributes_dict = json.load(f)

# Load CoOp models from test_eval_prompts
# Let's define the 4 zero-shot prompt templates
p1 = [f"{c}." for c in class_names]
p2 = [f"a photo of {c}." for c in class_names]
p3 = [f"a photo of a {c}." for c in class_names]
p4 = []
for c in class_names:
    k = c.lower().replace(' ', '_')
    d = attributes_dict.get(k, {})
    sp = d.get('species', 'pet')
    p4.append(f"a photo of a {c}, a type of {sp}.")

def get_per_class(prompts):
    toks = clip_processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        t_f = clip_model.get_text_features(**toks)
        t_f = t_f / t_f.norm(dim=-1, keepdim=True)
        logits = (test_features_raw @ t_f.t()) * clip_model.logit_scale.exp()
        preds = logits.argmax(dim=-1)
    accs = {}
    for c in range(num_classes):
        mask = (test_labels == c)
        accs[c] = (preds[mask] == test_labels[mask]).float().mean().item() * 100.0
    tot = (preds == test_labels).float().mean().item() * 100.0
    return accs, tot

a1, t1 = get_per_class(p1)
a2, t2 = get_per_class(p2)
a3, t3 = get_per_class(p3)
a4, t4 = get_per_class(p4)

with open("test_eval_results.json") as f:
    prev = json.load(f)

coop_b = prev["coop_base_acc"]
coop_g = prev["coop_gemma_acc"]

res = {}
target_classes = ["Bengal", "Beagle", "Great Pyrenees", "Ragdoll", "Samoyed", "Birman", "Persian"]
for name in target_classes:
    idx = class_names.index(name)
    res[name] = {
        "p1": round(a1[idx], 2),
        "p2": round(a2[idx], 2),
        "p3": round(a3[idx], 2),
        "p4": round(a4[idx], 2),
        "coop_base": round(coop_b[str(idx)], 2),
        "coop_gemma": round(coop_g[str(idx)], 2),
    }

print(json.dumps(res, indent=2))
with open("exact_6_prompts_results.json", "w") as f:
    json.dump(res, f, indent=2)
