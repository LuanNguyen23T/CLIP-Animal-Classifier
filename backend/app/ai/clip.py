"""
CLIPZeroShotClassifier — Production implementation for CLIP Pet Breed Classification.

Architecture:
  - Loads openai/clip-vit-base-patch32 once (singleton-style, lazy init)
  - Builds text prompts per breed using the "a photo of a {breed}, a type of {species}" template
  - Runs image through CLIP image encoder, computes cosine similarity with all text embeddings
  - Applies softmax to get probability distribution (top-K results)
  - Supports optional Test-Time Augmentation (TTA) for ~0.5% accuracy improvement

Model version tracking:
  - "clip-vit-base-patch32-zero-shot"      → standard inference
  - "clip-vit-base-patch32-zero-shot-tta"  → with TTA (5 augmented crops)

Accuracy (from R&D benchmark):
  - Zero-Shot:     80.16% on Full Test set
  - Zero-Shot TTA: 80.65% on Full Test set
"""

from __future__ import annotations

import logging
from typing import List, Union, Optional
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from transformers import CLIPModel, CLIPProcessor

from app.ai.base import Classifier, ClassifierResult, PredictionResultItem

logger = logging.getLogger(__name__)

# ─── Breed catalogue — matches DB (id 1-37, 0-indexed internally) ────────────
# Format: (breed_id: int, breed_name: str, species: str, display_name: str)
# display_name is used for text prompts (human-readable, no underscores)
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

# Prompt templates — ensemble of prompts (averaged embeddings) improves accuracy
PROMPT_TEMPLATES = [
    "a photo of a {breed}",
    "a photo of a {breed}, a type of pet",
    "a close up photo of a {breed}",
    "a photo of a {breed}, a type of {species}",
]

MODEL_ID = "openai/clip-vit-base-patch32"

# TTA augmentation configs (offsets as fraction of image size)
TTA_CROPS = [
    (0.0, 0.0, 1.0, 1.0),    # full image
    (0.0, 0.0, 0.85, 0.85),  # top-left crop
    (0.15, 0.0, 1.0, 0.85),  # top-right crop
    (0.0, 0.15, 0.85, 1.0),  # bottom-left crop
    (0.15, 0.15, 1.0, 1.0),  # bottom-right crop
]


def _build_text_prompts() -> List[str]:
    """Build prompt strings for all 37 breeds using template ensemble."""
    prompts = []
    for _, _, species, display_name in BREED_CATALOGUE:
        # Average multiple templates per breed
        for tmpl in PROMPT_TEMPLATES:
            prompts.append(tmpl.format(breed=display_name, species=species.lower()))
    return prompts


class CLIPZeroShotClassifier(Classifier):
    """
    Zero-Shot CLIP classifier for Oxford-IIIT Pet breeds.

    Args:
        use_tta: If True, apply 5-crop Test-Time Augmentation.
        top_k: Number of top results to return (default 5).
        device: 'cpu', 'cuda', or None for auto-detect.
    """

    def __init__(
        self,
        use_tta: bool = False,
        top_k: int = 5,
        device: Optional[str] = None,
    ):
        self.use_tta = use_tta
        self.top_k = top_k

        # Auto-detect device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model_version = (
            "clip-vit-base-patch32-zero-shot-tta"
            if use_tta
            else "clip-vit-base-patch32-zero-shot"
        )

        logger.info(f"[CLIP] Loading {MODEL_ID} on {self.device}...")
        self._model = CLIPModel.from_pretrained(MODEL_ID).to(self.device)
        self._processor = CLIPProcessor.from_pretrained(MODEL_ID)
        self._model.eval()

        # Pre-compute and cache text embeddings (only done once at startup)
        logger.info("[CLIP] Building text embeddings for 37 breeds...")
        self._text_embeddings = self._build_text_embeddings()
        logger.info("[CLIP] Ready.")

    def _build_text_embeddings(self) -> torch.Tensor:
        """
        Pre-compute averaged text embeddings for all breeds.
        Returns tensor of shape [37, embed_dim], L2-normalised.
        """
        prompts = _build_text_prompts()
        n_templates = len(PROMPT_TEMPLATES)
        n_breeds = len(BREED_CATALOGUE)

        # Tokenize all prompts at once
        inputs = self._processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self._model.get_text_features(**inputs)
            # transformers >=4.50 may return ModelOutput instead of tensor
            text_feats = out if isinstance(out, torch.Tensor) else out.pooler_output
            text_feats = F.normalize(text_feats, dim=-1)

        # Reshape to [37, n_templates, embed_dim] and average across templates
        text_feats = text_feats.view(n_breeds, n_templates, -1)
        text_feats = text_feats.mean(dim=1)
        text_feats = F.normalize(text_feats, dim=-1)  # re-normalise after averaging

        return text_feats  # [37, 512]

    def _get_image_embedding(self, image: PILImage.Image) -> torch.Tensor:
        """Encode a single PIL image, return L2-normalised embedding [1, 512]."""
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.get_image_features(**inputs)
            # transformers >=4.50 may return ModelOutput instead of tensor
            img_feats = out if isinstance(out, torch.Tensor) else out.pooler_output
            img_feats = F.normalize(img_feats, dim=-1)
        return img_feats  # [1, 512]

    def _get_tta_embedding(self, image: PILImage.Image) -> torch.Tensor:
        """
        5-crop TTA: average embeddings from full image + 4 corner crops.
        Returns L2-normalised averaged embedding [1, 512].
        """
        w, h = image.size
        embeddings = []

        for (x0, y0, x1, y1) in TTA_CROPS:
            crop = image.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
            embeddings.append(self._get_image_embedding(crop))

        stacked = torch.cat(embeddings, dim=0)   # [5, 512]
        averaged = stacked.mean(dim=0, keepdim=True)   # [1, 512]
        return F.normalize(averaged, dim=-1)

    def _load_image(self, image: Union[str, bytes, PILImage.Image]) -> PILImage.Image:
        """Accept file path, raw bytes, or PIL Image — return PIL Image in RGB."""
        if isinstance(image, str):
            return PILImage.open(image).convert("RGB")
        elif isinstance(image, bytes):
            import io
            return PILImage.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, PILImage.Image):
            return image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def predict(self, image: Union[str, bytes, PILImage.Image]) -> ClassifierResult:
        """
        Run zero-shot CLIP classification.

        Returns ClassifierResult with predicted_breed_id matching DB IDs (1-37).
        """
        pil_image = self._load_image(image)

        # Get image embedding
        if self.use_tta:
            img_embedding = self._get_tta_embedding(pil_image)
        else:
            img_embedding = self._get_image_embedding(pil_image)

        # Cosine similarity: [1, 37]
        logits = img_embedding @ self._text_embeddings.T  # [1, 37]
        logits = logits.squeeze(0)  # [37]

        # Temperature-scaled softmax (CLIP default temp = 100 * logit_scale)
        # We use softmax directly on cosine similarities for stability
        probs = F.softmax(logits * 100.0, dim=0)  # [37]

        # Top-K results
        top_k_count = min(self.top_k, len(BREED_CATALOGUE))
        top_probs, top_indices = torch.topk(probs, k=top_k_count)

        top_k_results: List[PredictionResultItem] = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            breed_id, breed_name, species, _ = BREED_CATALOGUE[idx]
            top_k_results.append(PredictionResultItem(
                breed_id=breed_id,
                breed_name=breed_name,
                species=species,
                score=round(prob, 6),
            ))

        best = top_k_results[0]
        return ClassifierResult(
            predicted_breed_id=best.breed_id,
            score=round(best.score, 6),
            top_k_results=top_k_results,
            model_version=self.model_version,
        )
