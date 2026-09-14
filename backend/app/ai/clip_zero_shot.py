"""
clip_zero_shot.py — CLIP Zero-Shot Baseline classifier using shared backbone.

Pipeline: image → [TTA 5-view] → CLIP image encoder → cosine similarity
          with enriched text embeddings → softmax → top-K results

Accuracy (benchmark Full Trainval + TTA 5-view): 87.76%
"""
from __future__ import annotations

import io
import logging
from typing import List, Optional, Union
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image as PILImage

from app.ai.base import Classifier, ClassifierResult, PredictionResultItem
from app.ai.shared_backbone import (
    BREED_CATALOGUE, get_clip_backbone, get_text_features
)

logger = logging.getLogger(__name__)

# TTA 5-crop offsets (fraction of image dimensions)
TTA_CROPS = [
    (0.0,  0.0,  1.0,  1.0),   # full image
    (0.0,  0.0,  0.85, 0.85),  # top-left
    (0.15, 0.0,  1.0,  0.85),  # top-right
    (0.0,  0.15, 0.85, 1.0),   # bottom-left
    (0.15, 0.15, 1.0,  1.0),   # bottom-right
]

MODEL_VERSION = "clip-vit-b32-zero-shot-tta"


class ZeroShotClassifier(Classifier):
    """
    CLIP Zero-Shot Baseline classifier.
    No training required — pure prompt-based classification.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self._model, self._processor, self._device = get_clip_backbone()
        self._text_features = get_text_features()  # [37, 512]
        logger.info("[ZeroShot] Ready.")

    def _load_image(self, image: Union[str, bytes, PILImage.Image]) -> PILImage.Image:
        if isinstance(image, str):
            return PILImage.open(image).convert("RGB")
        elif isinstance(image, bytes):
            return PILImage.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, PILImage.Image):
            return image.convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _encode_image(self, img: PILImage.Image) -> torch.Tensor:
        """Encode a single PIL image → L2-normalized [1, 512]."""
        inputs = self._processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.get_image_features(**inputs)
            feats = out if isinstance(out, torch.Tensor) else out.pooler_output
            return F.normalize(feats, dim=-1)

    def _encode_tta(self, img: PILImage.Image) -> torch.Tensor:
        """5-crop TTA → average L2-normalized embedding [1, 512]."""
        w, h = img.size
        embeddings = []
        for x0, y0, x1, y1 in TTA_CROPS:
            crop = img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
            embeddings.append(self._encode_image(crop))
        stacked = torch.cat(embeddings, dim=0)   # [5, 512]
        return F.normalize(stacked.mean(dim=0, keepdim=True), dim=-1)

    def predict(self, image: Union[str, bytes, PILImage.Image]) -> ClassifierResult:
        pil = self._load_image(image)
        img_feat = self._encode_tta(pil)  # [1, 512]

        logits = (img_feat @ self._text_features.T).squeeze(0)  # [37]
        probs = F.softmax(logits * 100.0, dim=0)

        k = min(self.top_k, len(BREED_CATALOGUE))
        top_probs, top_idx = torch.topk(probs, k=k)

        top_k: List[PredictionResultItem] = []
        for prob, idx in zip(top_probs.tolist(), top_idx.tolist()):
            breed_id, breed_name, species, _ = BREED_CATALOGUE[idx]
            top_k.append(PredictionResultItem(
                breed_id=breed_id, breed_name=breed_name,
                species=species, score=round(prob, 6)
            ))

        best = top_k[0]
        return ClassifierResult(
            predicted_breed_id=best.breed_id,
            score=round(best.score, 6),
            top_k_results=top_k,
            model_version=MODEL_VERSION,
        )
