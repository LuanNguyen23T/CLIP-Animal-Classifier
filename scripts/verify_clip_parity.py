"""Verify deployed CLIP artifacts against the benchmark's cached TTA test set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

from export_clip_weights import (
    CoOpClassifier,
    EXPECTED_TTA_ACCURACY,
    MODEL_ID,
    MODEL_REVISION,
    NUM_CLASSES,
    OXFORD_CLASSES,
    construct_coop_input_ids,
    projected_features,
)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == labels).float().mean().item() * 100


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=root / "dataset")
    parser.add_argument("--test-features", type=Path)
    parser.add_argument("--test-labels", type=Path)
    parser.add_argument("--weights-dir", type=Path, default=root / "backend" / "weights")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tolerance", type=float, default=0.015)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    features_path = args.test_features or args.dataset_dir / "cached" / "test_features_tta.pt"
    labels_path = args.test_labels or args.dataset_dir / "test_labels.pt"
    features = torch.load(
        features_path,
        map_location=device,
        weights_only=False,
    ).float()
    labels = torch.load(
        labels_path, map_location=device, weights_only=False
    ).long()
    text_features = torch.load(
        args.weights_dir / "text_features_enr.pt", map_location=device, weights_only=False
    ).float()

    model_kwargs = {
        "revision": MODEL_REVISION,
        "local_files_only": args.local_files_only,
    }
    if args.local_files_only:
        model_kwargs["use_safetensors"] = False
    clip_model = CLIPModel.from_pretrained(MODEL_ID, **model_kwargs).to(device)
    processor = CLIPProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=args.local_files_only
    )
    logit_scale = clip_model.logit_scale.exp().item()

    results = {"zero_shot": accuracy(features @ text_features.t(), labels)}

    cache = torch.load(
        args.weights_dir / "tip_adapter_cache.pt", map_location=device, weights_only=False
    )
    support = F.normalize(cache["W_1"].float(), dim=-1)
    one_hot = F.one_hot(cache["labels"].long(), num_classes=NUM_CLASSES).float()
    affinity = torch.exp(-10.0 * (1.0 - features @ support.t()))
    cache_logits = (
        (affinity @ one_hot) / (len(support) / float(NUM_CLASSES))
    ) * logit_scale
    zero_shot_logits = (features @ text_features.t()) * logit_scale
    results["tip_adapter"] = accuracy(zero_shot_logits + cache_logits, labels)

    tip_f = torch.load(
        args.weights_dir / "tip_adapter_f_full.pt", map_location=device, weights_only=False
    )
    tip_support = F.normalize(tip_f["W_1"].float(), dim=-1)
    tip_affinity = torch.exp(-10.0 * (1.0 - features @ tip_support.t()))
    tip_logits = (features @ tip_f["W_text"].float().t()) * logit_scale
    tip_logits += (
        (tip_affinity @ tip_f["W_2"].float()) / tip_f["K_avg"].float()
    ) * logit_scale
    results["tip_adapter_f"] = accuracy(tip_logits, labels)

    adapter = torch.load(
        args.weights_dir / "clip_adapter_full.pt", map_location=device, weights_only=False
    )
    hidden = F.relu(F.linear(features, adapter["mlp.0.weight"], adapter["mlp.0.bias"]))
    residual = F.linear(hidden, adapter["mlp.2.weight"], adapter["mlp.2.bias"])
    adapted = F.normalize(0.2 * residual + 0.8 * features, dim=-1)
    results["clip_adapter"] = accuracy(
        (adapted @ text_features.t()) * logit_scale, labels
    )

    coop = CoOpClassifier(clip_model).to(device)
    context = torch.load(
        args.weights_dir / "coop_context_full.pt", map_location=device, weights_only=False
    ).float()
    coop.ctx.data.copy_(context)
    input_ids = construct_coop_input_ids(OXFORD_CLASSES, processor).to(device)
    with torch.no_grad():
        coop_logits = coop(features, input_ids)
    results["coop"] = accuracy(coop_logits, labels)
    coop.restore_embedding()

    failures = []
    print("CLIP benchmark parity (TTA 5-view, Full Trainval)")
    for name, expected in EXPECTED_TTA_ACCURACY.items():
        actual = results[name]
        passed = abs(actual - expected) <= args.tolerance
        print(f"  {name:16s} expected={expected:6.2f}% actual={actual:8.4f}%  {'PASS' if passed else 'FAIL'}")
        if not passed:
            failures.append(name)

    report = {
        "scenario": "TTA 5-view (Full Trainval)",
        "tolerance": args.tolerance,
        "expected": EXPECTED_TTA_ACCURACY,
        "actual": results,
        "passed": not failures,
    }
    report_path = args.weights_dir / "clip_parity_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if failures:
        raise SystemExit(f"Parity failed for: {', '.join(failures)}")
    print(f"All CLIP modules match the benchmark. Report: {report_path}")


if __name__ == "__main__":
    main()
