"""
Reproducible evaluation script for CuPL and LaBo LLM-guided vision baselines
on Oxford-IIIT Pet and Stanford Dogs official test sets using CLIP ViT-B/32.
"""

import os
import json
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Evaluating CuPL and LaBo on compute device: {device}")

RESULTS_DIR = "Semester/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

model_id = "openai/clip-vit-base-patch32"
tokenizer = CLIPTokenizer.from_pretrained(model_id)
clip_model = CLIPModel.from_pretrained(model_id).to(device)
clip_model.eval()

# ---------------------------------------------------------
# 1. Oxford-IIIT Pet
# ---------------------------------------------------------
with open("Semester/cached_features/oxford_pets/classes.json") as f:
    ox_classes = json.load(f)
with open("Semester/attributes/oxford_pets_attributes.json") as f:
    ox_attr = json.load(f)

ox_test_clip = torch.load("Semester/cached_features/oxford_pets/test_clip.pt", map_location=device).float()
ox_test_tta = torch.load("dataset/cached/test_features_tta.pt", map_location=device).float()
ox_test_labels = torch.load("Semester/cached_features/oxford_pets/test_labels.pt", map_location=device)

cupl_templates_ox = [
    "a photo of a {name}, which has {head_and_ears}.",
    "a photo of a {name} with {coat}.",
    "a photo of a {name} displaying {distinguishing_mark}.",
    "a photo of a {name}, a pet with {body}.",
    "a close-up photo of a {name} showing its {eyes}."
]

w_cupl_ox = []
w_labo_ox = []
for c in ox_classes:
    k = c.lower().replace(" ", "_")
    attr = ox_attr[k]["attributes"]
    prompts_c = [t.format(name=c.replace("_", " "), **attr) for t in cupl_templates_ox]
    inps_c = tokenizer(prompts_c, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
    with torch.no_grad():
        tf_c = F.normalize(clip_model.get_text_features(**inps_c), dim=-1)
        w_cupl_ox.append(F.normalize(tf_c.mean(dim=0, keepdim=True), dim=-1))
        
    prompts_l = [
        f"{c.replace('_', ' ')} has {attr['head_and_ears']}",
        f"{c.replace('_', ' ')} has {attr['eyes']}",
        f"{c.replace('_', ' ')} has {attr['coat']}",
        f"{c.replace('_', ' ')} has {attr['body']}",
        f"{c.replace('_', ' ')} has {attr['distinguishing_mark']}"
    ]
    inps_l = tokenizer(prompts_l, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
    with torch.no_grad():
        tf_l = F.normalize(clip_model.get_text_features(**inps_l), dim=-1)
        w_labo_ox.append(tf_l)

W_cupl_ox = torch.cat(w_cupl_ox, dim=0) # [37, 512]
W_labo_ox = torch.stack(w_labo_ox, dim=0) # [37, 5, 512]

with torch.no_grad():
    acc_cupl_ox_raw = ((F.normalize(ox_test_clip, dim=-1) @ W_cupl_ox.t()).argmax(-1) == ox_test_labels).float().mean().item() * 100.0
    acc_cupl_ox_tta = ((F.normalize(ox_test_tta, dim=-1) @ W_cupl_ox.t()).argmax(-1) == ox_test_labels).float().mean().item() * 100.0
    correct_cupl_ox = ((F.normalize(ox_test_tta, dim=-1) @ W_cupl_ox.t()).argmax(-1) == ox_test_labels).sum().item()

    sim_labo_raw = torch.einsum("nd,ckd->nck", F.normalize(ox_test_clip, dim=-1), W_labo_ox).mean(dim=-1)
    acc_labo_ox_raw = (sim_labo_raw.argmax(-1) == ox_test_labels).float().mean().item() * 100.0
    sim_labo_tta = torch.einsum("nd,ckd->nck", F.normalize(ox_test_tta, dim=-1), W_labo_ox).mean(dim=-1)
    acc_labo_ox_tta = (sim_labo_tta.argmax(-1) == ox_test_labels).float().mean().item() * 100.0
    correct_labo_ox = (sim_labo_tta.argmax(-1) == ox_test_labels).sum().item()

# ---------------------------------------------------------
# 2. Stanford Dogs
# ---------------------------------------------------------
with open("Semester/cached_features/stanford_dogs/classes.json") as f:
    dogs_classes = json.load(f)
with open("Semester/attributes/stanford_dogs_attributes.json") as f:
    dogs_attr = json.load(f)

dogs_test_clip = torch.load("Semester/cached_features/stanford_dogs/test_clip.pt", map_location=device).float()
dogs_test_labels = torch.load("Semester/cached_features/stanford_dogs/test_labels.pt", map_location=device)

cupl_templates_dogs = [
    "a photo of a {name}, which has {head_and_ears}.",
    "a photo of a {name} with {coat}.",
    "a photo of a {name} displaying {distinguishing_mark}.",
    "a photo of a {name}, a dog with {body}.",
    "a close-up photo of a {name} showing its {eyes}."
]

w_cupl_dogs = []
w_labo_dogs = []
for c in dogs_classes:
    attr = dogs_attr[c]["attributes"]
    prompts_c = [t.format(name=c, **attr) for t in cupl_templates_dogs]
    inps_c = tokenizer(prompts_c, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
    with torch.no_grad():
        tf_c = F.normalize(clip_model.get_text_features(**inps_c), dim=-1)
        w_cupl_dogs.append(F.normalize(tf_c.mean(dim=0, keepdim=True), dim=-1))
        
    prompts_l = [
        f"{c} has {attr['head_and_ears']}",
        f"{c} has {attr['eyes']}",
        f"{c} has {attr['coat']}",
        f"{c} has {attr['body']}",
        f"{c} has {attr['distinguishing_mark']}"
    ]
    inps_l = tokenizer(prompts_l, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
    with torch.no_grad():
        tf_l = F.normalize(clip_model.get_text_features(**inps_l), dim=-1)
        w_labo_dogs.append(tf_l)

W_cupl_dogs = torch.cat(w_cupl_dogs, dim=0) # [120, 512]
W_labo_dogs = torch.stack(w_labo_dogs, dim=0) # [120, 5, 512]

with torch.no_grad():
    dogs_norm = F.normalize(dogs_test_clip, dim=-1)
    acc_cupl_dogs = ((dogs_norm @ W_cupl_dogs.t()).argmax(-1) == dogs_test_labels).float().mean().item() * 100.0
    correct_cupl_dogs = ((dogs_norm @ W_cupl_dogs.t()).argmax(-1) == dogs_test_labels).sum().item()

    sim_labo_dogs = torch.einsum("nd,ckd->nck", dogs_norm, W_labo_dogs).mean(dim=-1)
    acc_labo_dogs = (sim_labo_dogs.argmax(-1) == dogs_test_labels).float().mean().item() * 100.0
    correct_labo_dogs = (sim_labo_dogs.argmax(-1) == dogs_test_labels).sum().item()

results = {
    "oxford_iiit_pet": {
        "cupl": {
            "test_raw_acc": acc_cupl_ox_raw,
            "test_tta_acc": acc_cupl_ox_tta,
            "correct": correct_cupl_ox,
            "total": len(ox_test_labels)
        },
        "labo": {
            "test_raw_acc": acc_labo_ox_raw,
            "test_tta_acc": acc_labo_ox_tta,
            "correct": correct_labo_ox,
            "total": len(ox_test_labels)
        }
    },
    "stanford_dogs": {
        "cupl": {
            "test_top1_acc": acc_cupl_dogs,
            "correct": correct_cupl_dogs,
            "total": len(dogs_test_labels)
        },
        "labo": {
            "test_top1_acc": acc_labo_dogs,
            "correct": correct_labo_dogs,
            "total": len(dogs_test_labels)
        }
    }
}

out_path = os.path.join(RESULTS_DIR, "verified_cupl_labo_baselines.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print("\n=== VERIFIED BASELINE BENCHMARK SUMMARY ===")
print(f"Oxford Pets - CuPL: {acc_cupl_ox_tta:.2f}% TTA ({correct_cupl_ox}/{len(ox_test_labels)})")
print(f"Oxford Pets - LaBo: {acc_labo_ox_tta:.2f}% TTA ({correct_labo_ox}/{len(ox_test_labels)})")
print(f"Stanford Dogs - CuPL: {acc_cupl_dogs:.2f}% ({correct_cupl_dogs}/{len(dogs_test_labels)})")
print(f"Stanford Dogs - LaBo: {acc_labo_dogs:.2f}% ({correct_labo_dogs}/{len(dogs_test_labels)})")
print(f"Saved results to: {out_path}")

