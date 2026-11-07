"""
Script to evaluate official CuPL prompts (mapped from ImageNet synsets)
on the Stanford Dogs official test split (8,580 images) using CLIP ViT-B/32.

Data sources:
- Prompts: sarahpratt/CuPL official imagenet_prompts_full.json
- Mapping: synset ID mapping via sense_to_idx.py & imagenet_classes.py
- Test features: Semester/cached_features/stanford_dogs_official/test_clip.pt
"""

import json
import os
import urllib.request
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer
from datasets import load_dataset

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Download official CuPL prompts & ImageNet class mappings
    print("Downloading CuPL official ImageNet prompts...")
    url_prompts = 'https://raw.githubusercontent.com/sarahpratt/CuPL/main/all_prompts/full_prompts/imagenet_prompts_full.json'
    req = urllib.request.Request(url_prompts, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp:
        cupl_prompts = json.load(resp)

    url_sense = 'https://raw.githubusercontent.com/sarahpratt/CuPL/main/imagenet_classnames/sense_to_idx.py'
    req = urllib.request.Request(url_sense, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp:
        code_sense = resp.read().decode()

    url_classes = 'https://raw.githubusercontent.com/sarahpratt/CuPL/main/imagenet_classnames/imagenet_classes.py'
    req = urllib.request.Request(url_classes, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp:
        code_classes = resp.read().decode()

    scope = {}
    exec(code_sense, scope)
    exec(code_classes, scope)
    sense_to_name = {'n' + scope['sense'][item]['id'].split('-')[0]: scope['imagenet_classes'][i] for i, item in enumerate(scope['sense'])}

    # 2. Get Stanford Dogs class names & map to ImageNet text names
    ds = load_dataset('maurice-fp/stanford-dogs', split='train')
    raw_class_names = ds.features['label'].names
    print(f"Total Stanford Dogs classes: {len(raw_class_names)}")

    # 3. Load pre-cached CLIP test features
    feat_dir = 'Semester/cached_features/stanford_dogs_official'
    test_clip = torch.load(f'{feat_dir}/test_clip.pt', map_location=device).float()
    test_labels = torch.load(f'{feat_dir}/test_labels.pt', map_location=device)
    print(f"Loaded {test_clip.shape[0]} test samples.")

    # 4. Encode CuPL prompts with CLIP ViT-B/32
    model_id = 'openai/clip-vit-base-patch32'
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    clip_model = CLIPModel.from_pretrained(model_id).to(device)
    clip_model.eval()

    templates_handcrafted = [
        'a photo of a {c}, a type of dog.',
        'a photo of the {c}.',
        'a photo of a clean {c}.',
        'a close-up photo of a {c}.',
        'a cropped photo of the {c}.',
        'a photo of my {c}.'
    ]

    cupl_weights_list = []
    cupl_comb_list = []

    for raw in raw_class_names:
        synset = raw.split('-')[0]
        textname = sense_to_name[synset]
        prompts = cupl_prompts[textname]

        # Standalone CuPL text weights
        inps = tokenizer(prompts, padding=True, truncation=True, max_length=77, return_tensors='pt').to(device)
        with torch.no_grad():
            tf = F.normalize(clip_model.get_text_features(**inps), dim=-1)
            w_c = F.normalize(tf.mean(dim=0, keepdim=True), dim=-1)
            cupl_weights_list.append(w_c)

        # Combined CuPL + Handcrafted templates
        clean_name = textname.replace('_', ' ')
        templated = [t.format(c=clean_name) for t in templates_handcrafted]
        all_p = prompts + templated
        inps_comb = tokenizer(all_p, padding=True, truncation=True, max_length=77, return_tensors='pt').to(device)
        with torch.no_grad():
            tf_comb = F.normalize(clip_model.get_text_features(**inps_comb), dim=-1)
            w_comb = F.normalize(tf_comb.mean(dim=0, keepdim=True), dim=-1)
            cupl_comb_list.append(w_comb)

    W_cupl = torch.cat(cupl_weights_list, dim=0)
    W_comb = torch.cat(cupl_comb_list, dim=0)

    # 5. Evaluate on test set
    logits_cupl = test_clip @ W_cupl.T
    preds_cupl = logits_cupl.argmax(dim=-1)
    acc_cupl = (preds_cupl == test_labels).float().mean().item() * 100
    corr_cupl = (preds_cupl == test_labels).sum().item()

    logits_comb = test_clip @ W_comb.T
    preds_comb = logits_comb.argmax(dim=-1)
    acc_comb = (preds_comb == test_labels).float().mean().item() * 100
    corr_comb = (preds_comb == test_labels).sum().item()

    print(f"CuPL Standalone Top-1 Acc: {acc_cupl:.2f}% ({corr_cupl}/{test_clip.shape[0]})")
    print(f"CuPL + Handcrafted Combined Top-1 Acc: {acc_comb:.2f}% ({corr_comb}/{test_clip.shape[0]})")

    result = {
        "dataset": "Stanford Dogs Official Test Split (8,580 images)",
        "model": "openai/clip-vit-base-patch32",
        "cupl_source": "sarahpratt/CuPL imagenet_prompts_full.json mapped via synset",
        "cupl_standalone": {
            "top1_acc": acc_cupl,
            "correct": corr_cupl,
            "total": test_clip.shape[0]
        },
        "cupl_combined_templates": {
            "top1_acc": acc_comb,
            "correct": corr_comb,
            "total": test_clip.shape[0]
        }
    }

    os.makedirs("Semester/results", exist_ok=True)
    out_path = "Semester/results/verified_cupl_official_stanford_dogs.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved results to {out_path}")

if __name__ == '__main__':
    main()

