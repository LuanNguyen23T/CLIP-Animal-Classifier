"""Export the CLIP artifacts used by the benchmark notebook.

The seed, split, training order, optimizers, and hyperparameters mirror
``notebooks/benchmark_clip_animal_classifier.ipynb``. Paths are configurable
so the same script can run locally or on Kaggle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.datasets import OxfordIIITPet
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor


SEED = 42
NUM_CLASSES = 37
MODEL_ID = "openai/clip-vit-base-patch32"
MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
PROMPT_TEMPLATE = "Picture of a pet with breed: {c}, which is a type of animal."
EXPECTED_TTA_ACCURACY = {
    "zero_shot": 87.76,
    "tip_adapter": 87.90,
    "tip_adapter_f": 89.56,
    "clip_adapter": 90.27,
    "coop": 91.61,
}

OXFORD_CLASSES = [
    "Abyssinian", "American Bulldog", "American Pit Bull Terrier", "Basset Hound",
    "Beagle", "Bengal", "Birman", "Bombay", "Boxer", "British Shorthair",
    "Chihuahua", "Egyptian Mau", "English Cocker Spaniel", "English Setter",
    "German Shorthaired", "Great Pyrenees", "Havanese", "Japanese Chin",
    "Keeshond", "Leonberger", "Maine Coon", "Miniature Pinscher", "Newfoundland",
    "Persian", "Pomeranian", "Pug", "Ragdoll", "Russian Blue", "Saint Bernard",
    "Samoyed", "Scottish Terrier", "Shiba Inu", "Siamese", "Sphynx",
    "Staffordshire Bull Terrier", "Wheaten Terrier", "Yorkshire Terrier",
]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def projected_features(output) -> torch.Tensor:
    """Unwrap CLIPModel.get_*_features without applying projection twice."""
    if isinstance(output, torch.Tensor):
        return output
    for attribute in ("text_embeds", "image_embeds", "pooler_output"):
        value = getattr(output, attribute, None)
        if isinstance(value, torch.Tensor):
            return value
    raise TypeError(f"Unsupported CLIP feature output: {type(output)!r}")


class CLIPTransformDataset(Dataset):
    def __init__(self, base_dataset, processor):
        self.base_dataset = base_dataset
        self.processor = processor

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, label = self.base_dataset[index]
        pixels = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return pixels, label


def extract_trainval_features(dataset, model, processor, device):
    loader = DataLoader(CLIPTransformDataset(dataset, processor), batch_size=128, shuffle=False)
    features, labels = [], []
    with torch.no_grad():
        for pixels, batch_labels in tqdm(loader, desc="CLIP trainval features"):
            output = model.get_image_features(pixel_values=pixels.to(device))
            features.append(F.normalize(projected_features(output), dim=-1).cpu())
            labels.append(batch_labels.cpu())
    return torch.cat(features), torch.cat(labels)


def get_stratified_split(features, labels, train_ratio=0.9, seed=SEED):
    set_seed(seed)
    train_indices, validation_indices = [], []
    for class_index in range(NUM_CLASSES):
        indices = (labels == class_index).nonzero(as_tuple=True)[0]
        permutation = torch.randperm(len(indices))
        train_count = max(1, int(len(indices) * train_ratio))
        train_indices.append(indices[permutation[:train_count]])
        validation_indices.append(indices[permutation[train_count:]])
    train_index = torch.cat(train_indices)
    validation_index = torch.cat(validation_indices)
    return (
        (features[train_index], labels[train_index]),
        (features[validation_index], labels[validation_index]),
    )


def run_tip_adapter(
    test_features, text_features, support_features, support_labels,
    logit_scale, alpha=1.0, beta=10.0,
):
    support_norm = F.normalize(support_features, dim=-1)
    one_hot = F.one_hot(support_labels, num_classes=NUM_CLASSES).float()
    zero_shot = (test_features @ text_features.t()) * logit_scale
    affinity = torch.exp(-beta * (1.0 - test_features @ support_norm.t()))
    cache = ((affinity @ one_hot) / (len(support_features) / float(NUM_CLASSES))) * logit_scale
    return zero_shot + alpha * cache


def grid_search_tip_adapter(
    validation_features, validation_labels, text_features, support_features,
    support_labels, logit_scale,
):
    best_accuracy, best_alpha, best_beta = 0.0, 1.0, 5.0
    for alpha in [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        for beta in [1.0, 2.0, 3.0, 5.0, 10.0]:
            logits = run_tip_adapter(
                validation_features, text_features, support_features, support_labels,
                logit_scale, alpha, beta,
            )
            accuracy = (logits.argmax(-1) == validation_labels).float().mean().item() * 100
            if accuracy > best_accuracy:
                best_accuracy, best_alpha, best_beta = accuracy, alpha, beta
    return best_alpha, best_beta, best_accuracy


class TipAdapterF(nn.Module):
    def __init__(self, support_features, support_labels, text_features, logit_scale, alpha=1.0, beta=10.0):
        super().__init__()
        self.W_1 = nn.Parameter(support_features.clone())
        self.register_buffer("W_2", F.one_hot(support_labels, num_classes=NUM_CLASSES).float())
        self.register_buffer("W_text", text_features.clone())
        self.register_buffer("K_avg", torch.tensor(len(support_features) / float(NUM_CLASSES)))
        self.alpha = alpha
        self.beta = beta
        self.logit_scale = logit_scale

    def forward(self, features):
        zero_shot = (features @ self.W_text.t()) * self.logit_scale
        support_norm = F.normalize(self.W_1, dim=-1)
        affinity = torch.exp(-self.beta * (1.0 - features @ support_norm.t()))
        cache = ((affinity @ self.W_2) / self.K_avg) * self.logit_scale
        return zero_shot + self.alpha * cache


def train_tip_adapter_f(
    support_features, support_labels, validation_features, validation_labels,
    text_features, logit_scale, device, alpha, beta,
):
    model = TipAdapterF(
        support_features.to(device), support_labels.to(device), text_features.to(device),
        logit_scale, alpha, beta,
    ).to(device)
    optimizer = torch.optim.SGD([model.W_1], lr=0.001, momentum=0.9, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(support_features.to(device), support_labels.to(device)),
        batch_size=256, shuffle=True,
    )
    best_accuracy, best_state = 0.0, None
    for _ in tqdm(range(20), desc="Tip-Adapter-F"):
        model.train()
        for features, labels in loader:
            optimizer.zero_grad()
            F.cross_entropy(model(features), labels).backward()
            optimizer.step()
        if validation_features is not None:
            model.eval()
            with torch.no_grad():
                logits = model(validation_features.to(device))
                accuracy = (logits.argmax(-1) == validation_labels.to(device)).float().mean().item() * 100
            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


class CLIPAdapter(nn.Module):
    def __init__(self, input_dim=512, bottleneck_dim=64, alpha=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim), nn.ReLU(), nn.Linear(bottleneck_dim, input_dim)
        )
        self.alpha = alpha
        nn.init.constant_(self.mlp[2].weight, 0.0)
        nn.init.constant_(self.mlp[2].bias, 0.0)

    def forward(self, features):
        residual = self.mlp(features)
        return F.normalize(self.alpha * residual + (1.0 - self.alpha) * features, dim=-1)


class AdapterClassifier(nn.Module):
    def __init__(self, adapter, text_features, logit_scale):
        super().__init__()
        self.adapter = adapter
        self.register_buffer("text_features", text_features.clone())
        self.logit_scale = logit_scale

    def forward(self, features):
        return (self.adapter(features) @ self.text_features.t()) * self.logit_scale


def train_clip_adapter(
    support_features, support_labels, validation_features, validation_labels,
    text_features, logit_scale, device,
):
    model = AdapterClassifier(CLIPAdapter().to(device), text_features.to(device), logit_scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(support_features.to(device), support_labels.to(device)),
        batch_size=256, shuffle=True,
    )
    best_accuracy, best_state = 0.0, None
    for _ in tqdm(range(50), desc="CLIP-Adapter"):
        model.train()
        for features, labels in loader:
            optimizer.zero_grad()
            F.cross_entropy(model(features), labels).backward()
            optimizer.step()
        if validation_features is not None:
            model.eval()
            with torch.no_grad():
                logits = model(validation_features.to(device))
                accuracy = (logits.argmax(-1) == validation_labels.to(device)).float().mean().item() * 100
            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def construct_coop_input_ids(class_names, processor, num_ctx=4):
    sequences = []
    for name in class_names:
        tokens = processor.tokenizer(name, return_tensors="pt")["input_ids"][0]
        sequence = torch.cat([
            tokens[:1], torch.tensor([49400 + index for index in range(num_ctx)]),
            tokens[1:-1], tokens[-1:],
        ])
        if len(sequence) < 77:
            sequence = torch.cat([sequence, torch.zeros(77 - len(sequence), dtype=torch.long)])
        else:
            sequence = sequence[:77]
            sequence[-1] = tokens[-1]
        sequences.append(sequence)
    return torch.stack(sequences)


class CoOpClassifier(nn.Module):
    def __init__(self, clip_model, num_ctx=4):
        super().__init__()
        self.clip_model = clip_model
        self.num_ctx = num_ctx
        self.ctx = nn.Parameter(torch.randn(num_ctx, 512, device=next(clip_model.parameters()).device) * 0.02)
        self._original_embedding_forward = clip_model.text_model.embeddings.token_embedding.forward
        self._install_context()

    def _install_context(self):
        original = self._original_embedding_forward

        def context_embedding(input_ids, *args, **kwargs):
            embeddings = original(input_ids, *args, **kwargs).clone()
            for index in range(self.num_ctx):
                mask = input_ids == (49400 + index)
                if mask.any():
                    embeddings[mask] = self.ctx[index].to(embeddings.dtype)
            return embeddings

        self.clip_model.text_model.embeddings.token_embedding.forward = context_embedding

    def restore_embedding(self):
        self.clip_model.text_model.embeddings.token_embedding.forward = self._original_embedding_forward

    def forward(self, image_features, input_ids):
        output = self.clip_model.get_text_features(input_ids=input_ids)
        text_features = F.normalize(projected_features(output), dim=-1)
        return (image_features @ text_features.t()) * self.clip_model.logit_scale.exp()


def train_coop(
    clip_model, support_features, support_labels, validation_features,
    validation_labels, input_ids, device,
):
    model = CoOpClassifier(clip_model).to(device)
    optimizer = torch.optim.AdamW([model.ctx], lr=0.002, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(support_features.to(device), support_labels.to(device)),
        batch_size=256, shuffle=True,
    )
    best_accuracy, best_context = 0.0, None
    for _ in tqdm(range(50), desc="CoOp"):
        model.train()
        for features, labels in loader:
            optimizer.zero_grad()
            F.cross_entropy(model(features, input_ids), labels).backward()
            optimizer.step()
        if validation_features is not None:
            model.eval()
            with torch.no_grad():
                logits = model(validation_features.to(device), input_ids)
                accuracy = (logits.argmax(-1) == validation_labels.to(device)).float().mean().item() * 100
            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                best_context = model.ctx.detach().clone()
    if best_context is not None:
        model.ctx.data.copy_(best_context)
    return model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_kaggle_cache_dir() -> Path:
    matches = sorted(Path("/kaggle/input").rglob("trainval_features.pt"))
    if len(matches) == 1:
        return matches[0].parent
    return Path("/kaggle/input/clip-benchmark-cache")


def parse_args():
    script_file = globals().get("__file__")
    repository_root = (
        Path(script_file).resolve().parents[1]
        if script_file
        else Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
    )
    kaggle = Path("/kaggle/working").exists()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("/kaggle/working/dataset") if kaggle else repository_root / "dataset",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=default_kaggle_cache_dir() if kaggle else repository_root / "dataset",
    )
    parser.add_argument(
        "--weights-dir", type=Path,
        default=Path("/kaggle/working/weights") if kaggle else repository_root / "backend" / "weights",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    if script_file is None:
        # Jupyter injects its own command-line arguments and does not define
        # __file__ when this source is pasted directly into a notebook cell.
        return parser.parse_known_args()[0]
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    args.dataset_root.mkdir(parents=True, exist_ok=True)
    set_seed()

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset_root}")
    print(f"Feature cache: {args.cache_dir}")
    print(f"Weights: {args.weights_dir}")

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
    clip_model.eval()

    text_inputs = processor(
        text=[PROMPT_TEMPLATE.format(c=name) for name in OXFORD_CLASSES],
        return_tensors="pt", padding=True,
    ).to(device)
    with torch.no_grad():
        text_features = F.normalize(projected_features(clip_model.get_text_features(**text_inputs)), dim=-1)
    torch.save(text_features.cpu(), args.weights_dir / "text_features_enr.pt")

    features_path = args.cache_dir / "trainval_features.pt"
    labels_path = args.cache_dir / "trainval_labels.pt"
    if features_path.exists() and labels_path.exists():
        trainval_features = torch.load(features_path, map_location="cpu", weights_only=False).float()
        trainval_labels = torch.load(labels_path, map_location="cpu", weights_only=False).long()
    else:
        dataset = OxfordIIITPet(
            root=str(args.dataset_root), split="trainval", download=args.download
        )
        if list(dataset.classes) != OXFORD_CLASSES:
            raise RuntimeError("Oxford-IIIT Pet class order does not match the benchmark")
        trainval_features, trainval_labels = extract_trainval_features(dataset, clip_model, processor, device)
        torch.save(trainval_features, args.weights_dir / "trainval_features.pt")
        torch.save(trainval_labels, args.weights_dir / "trainval_labels.pt")

    if trainval_features.shape != (3680, 512) or trainval_labels.shape != (3680,):
        raise RuntimeError("Unexpected Oxford-IIIT Pet trainval feature shape")
    if not torch.isfinite(trainval_features).all():
        raise RuntimeError("Trainval feature cache contains NaN or infinity")
    max_norm_error = (trainval_features.norm(dim=-1) - 1.0).abs().max().item()
    if max_norm_error > 1e-4:
        raise RuntimeError(
            f"Trainval features are not benchmark-normalized (max norm error {max_norm_error:.3g})"
        )
    if trainval_labels.min().item() != 0 or trainval_labels.max().item() != NUM_CLASSES - 1:
        raise RuntimeError("Unexpected Oxford-IIIT Pet label range")

    logit_scale = clip_model.logit_scale.exp().item()

    (train_90, labels_90), (validation, validation_labels) = get_stratified_split(
        trainval_features, trainval_labels
    )
    alpha, beta, validation_accuracy = grid_search_tip_adapter(
        validation, validation_labels, text_features.cpu(), train_90, labels_90,
        logit_scale,
    )
    print(f"Tip-Adapter grid search: alpha={alpha}, beta={beta}, val={validation_accuracy:.2f}%")

    torch.save(
        {"W_1": trainval_features, "labels": trainval_labels},
        args.weights_dir / "tip_adapter_cache.pt",
    )
    # Mirror notebook order: train 9:1 first, then Full Trainval.
    train_tip_adapter_f(
        train_90, labels_90, validation, validation_labels,
        text_features.cpu(), logit_scale, device, alpha, beta,
    )
    tip_full = train_tip_adapter_f(
        trainval_features, trainval_labels, None, None,
        text_features.cpu(), logit_scale, device, alpha, beta,
    )
    torch.save(tip_full.state_dict(), args.weights_dir / "tip_adapter_f_full.pt")

    train_clip_adapter(
        train_90, labels_90, validation, validation_labels,
        text_features.cpu(), logit_scale, device,
    )
    adapter_full = train_clip_adapter(
        trainval_features, trainval_labels, None, None,
        text_features.cpu(), logit_scale, device,
    )
    torch.save(adapter_full.adapter.state_dict(), args.weights_dir / "clip_adapter_full.pt")

    coop_ids = construct_coop_input_ids(OXFORD_CLASSES, processor).to(device)
    coop_90 = train_coop(
        clip_model, train_90, labels_90, validation, validation_labels, coop_ids, device
    )
    # Keep the first embedding patch installed while constructing the Full
    # Trainval model, matching the notebook's execution order exactly.
    coop_full = train_coop(
        clip_model, trainval_features, trainval_labels, None, None, coop_ids, device
    )
    torch.save(coop_full.ctx.detach().cpu(), args.weights_dir / "coop_context_full.pt")
    coop_full.restore_embedding()
    coop_90.restore_embedding()

    artifacts = [
        "text_features_enr.pt", "tip_adapter_cache.pt", "tip_adapter_f_full.pt",
        "clip_adapter_full.pt", "coop_context_full.pt",
    ]
    manifest = {
        "seed": SEED,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "scenario": "TTA 5-view (Full Trainval)",
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": version("torchvision"),
            "transformers": version("transformers"),
            "numpy": np.__version__,
            "device": str(device),
        },
        "expected_accuracy": EXPECTED_TTA_ACCURACY,
        "artifacts": {name: sha256(args.weights_dir / name) for name in artifacts},
    }
    (args.weights_dir / "clip_benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("Export complete. Run scripts/verify_clip_parity.py before deployment.")


if __name__ == "__main__":
    main()
