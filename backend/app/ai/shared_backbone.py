"""
shared_backbone.py — Singleton CLIP ViT-B/32 backbone shared across all classifiers.

All four methods (Zero-Shot, CLIP-Adapter, Tip-Adapter-F, CoOp) load the same
CLIP model instance to avoid loading it 4 times (~400MB each).
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

MODEL_ID = "openai/clip-vit-base-patch32"

# Prompt templates — ensemble of 4 for enriched text embeddings
PROMPT_TEMPLATES = [
    "a photo of a {breed}",
    "a photo of a {breed}, a type of pet",
    "a close up photo of a {breed}",
    "a photo of a {breed}, a type of {species}",
]

# 37 breeds — order matches DB ids 1-37
BREED_CATALOGUE = [
    (1,  "Abyssinian",                "Cat", "Abyssinian cat"),
    (2,  "american_bulldog",          "Dog", "American Bulldog"),
    (3,  "american_pit_bull_terrier", "Dog", "American Pit Bull Terrier"),
    (4,  "basset_hound",              "Dog", "Basset Hound"),
    (5,  "beagle",                    "Dog", "Beagle"),
    (6,  "Bengal",                    "Cat", "Bengal cat"),
    (7,  "Birman",                    "Cat", "Birman cat"),
    (8,  "Bombay",                    "Cat", "Bombay cat"),
    (9,  "boxer",                     "Dog", "Boxer dog"),
    (10, "British_Shorthair",         "Cat", "British Shorthair cat"),
    (11, "chihuahua",                 "Dog", "Chihuahua"),
    (12, "Egyptian_Mau",              "Cat", "Egyptian Mau cat"),
    (13, "english_cocker_spaniel",    "Dog", "English Cocker Spaniel"),
    (14, "english_setter",            "Dog", "English Setter"),
    (15, "german_shorthaired",        "Dog", "German Shorthaired Pointer"),
    (16, "great_pyrenees",            "Dog", "Great Pyrenees"),
    (17, "havanese",                  "Dog", "Havanese"),
    (18, "japanese_chin",             "Dog", "Japanese Chin"),
    (19, "keeshond",                  "Dog", "Keeshond"),
    (20, "leonberger",                "Dog", "Leonberger"),
    (21, "Maine_Coon",                "Cat", "Maine Coon cat"),
    (22, "miniature_pinscher",        "Dog", "Miniature Pinscher"),
    (23, "newfoundland",              "Dog", "Newfoundland dog"),
    (24, "Persian",                   "Cat", "Persian cat"),
    (25, "pomeranian",                "Dog", "Pomeranian"),
    (26, "pug",                       "Dog", "Pug"),
    (27, "Ragdoll",                   "Cat", "Ragdoll cat"),
    (28, "Russian_Blue",              "Cat", "Russian Blue cat"),
    (29, "saint_bernard",             "Dog", "Saint Bernard"),
    (30, "samoyed",                   "Dog", "Samoyed"),
    (31, "scottish_terrier",          "Dog", "Scottish Terrier"),
    (32, "shiba_inu",                 "Dog", "Shiba Inu"),
    (33, "Siamese",                   "Cat", "Siamese cat"),
    (34, "Sphynx",                    "Cat", "Sphynx cat"),
    (35, "staffordshire_bull_terrier","Dog", "Staffordshire Bull Terrier"),
    (36, "wheaten_terrier",           "Dog", "Wheaten Terrier"),
    (37, "yorkshire_terrier",         "Dog", "Yorkshire Terrier"),
]

# ─── Singleton state ──────────────────────────────────────────────────────────
_clip_model: Optional[CLIPModel] = None
_clip_processor: Optional[CLIPProcessor] = None
_text_features_enr: Optional[torch.Tensor] = None  # [37, 512] enriched
_device: Optional[torch.device] = None


def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_clip_backbone() -> Tuple[CLIPModel, CLIPProcessor, torch.device]:
    """Return singleton CLIP model + processor, loading once on first call."""
    global _clip_model, _clip_processor, _device

    if _clip_model is None:
        _device = get_device()
        logger.info(f"[SharedBackbone] Loading {MODEL_ID} on {_device}...")
        _clip_model = CLIPModel.from_pretrained(MODEL_ID).to(_device)
        _clip_processor = CLIPProcessor.from_pretrained(MODEL_ID)
        _clip_model.eval()
        logger.info("[SharedBackbone] CLIP backbone ready.")

    return _clip_model, _clip_processor, _device


def get_text_features() -> torch.Tensor:
    """
    Return cached enriched text embeddings [37, 512] — L2 normalized.
    Built once from 4-template ensemble per breed.
    """
    global _text_features_enr

    if _text_features_enr is None:
        from pathlib import Path
        weights_pt = Path(__file__).parent.parent.parent / "weights" / "text_features_enr.pt"

        if weights_pt.exists():
            logger.info("[SharedBackbone] Loading precomputed text_features_enr.pt...")
            _text_features_enr = torch.load(weights_pt, map_location=get_device())
        else:
            logger.info("[SharedBackbone] Building text embeddings from scratch...")
            _text_features_enr = _build_text_features_from_scratch()

    return _text_features_enr


def _build_text_features_from_scratch() -> torch.Tensor:
    """Compute enriched text embeddings using 4-template ensemble."""
    model, processor, device = get_clip_backbone()

    prompts = []
    for _, _, species, display_name in BREED_CATALOGUE:
        for tmpl in PROMPT_TEMPLATES:
            prompts.append(tmpl.format(breed=display_name, species=species.lower()))

    inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.get_text_features(**inputs)
        text_feats = out if isinstance(out, torch.Tensor) else out.pooler_output
        text_feats = F.normalize(text_feats, dim=-1)

    n_breeds = len(BREED_CATALOGUE)
    n_templates = len(PROMPT_TEMPLATES)
    text_feats = text_feats.view(n_breeds, n_templates, -1).mean(dim=1)
    return F.normalize(text_feats, dim=-1)


def get_logit_scale() -> float:
    """Return CLIP logit scale (exp of learned parameter)."""
    model, _, _ = get_clip_backbone()
    return model.logit_scale.exp().item()
