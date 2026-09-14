"""
registry.py — Lazy-loading model registry for all 4 CLIP classifiers.

Usage:
    from app.ai.registry import ClassifierRegistry, ModelMethod
    classifier = ClassifierRegistry.get(ModelMethod.ZERO_SHOT)
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Dict

from app.ai.base import Classifier

logger = logging.getLogger(__name__)


class ModelMethod(str, Enum):
    ZERO_SHOT    = "zero-shot"
    TIP_ADAPTER_F = "tip-adapter-f"
    CLIP_ADAPTER = "clip-adapter"
    COOP         = "coop"


# Human-readable display names for the UI
MODEL_DISPLAY_INFO: Dict[str, dict] = {
    ModelMethod.ZERO_SHOT: {
        "name":        "CLIP Zero-Shot Baseline",
        "description": "Pure prompt-based classification. No training required.",
        "accuracy":    87.76,
        "latency_ms":  0.85,
        "params":      0,
        "tag":         "Training-Free",
    },
    ModelMethod.TIP_ADAPTER_F: {
        "name":        "Tip-Adapter-F (Cache Adapter)",
        "description": "Fine-tuned cache model blending zero-shot and nearest-neighbor logits.",
        "accuracy":    89.56,
        "latency_ms":  0.92,
        "params":      1884160,
        "tag":         "Fine-Tuned",
    },
    ModelMethod.CLIP_ADAPTER: {
        "name":        "CLIP-Adapter (Feature Adapter)",
        "description": "Residual bottleneck MLP adapter on image features with cosine head.",
        "accuracy":    90.27,
        "latency_ms":  0.89,
        "params":      66112,
        "tag":         "Lightweight",
    },
    ModelMethod.COOP: {
        "name":        "CoOp (Prompt Tuning)",
        "description": "Learnable context vectors injected into CLIP text prompts.",
        "accuracy":    91.61,
        "latency_ms":  0.95,
        "params":      2048,
        "tag":         "Best Accuracy",
    },
}


class ClassifierRegistry:
    """
    Lazy-loading singleton registry.
    Each classifier is loaded at most once and cached for reuse.
    """
    _cache: Dict[ModelMethod, Classifier] = {}

    @classmethod
    def get(cls, method: ModelMethod) -> Classifier:
        """Return the classifier for the given method, loading it if necessary."""
        if method not in cls._cache:
            logger.info(f"[Registry] Loading classifier: {method}")
            cls._cache[method] = cls._load(method)
            logger.info(f"[Registry] Classifier ready: {method}")
        return cls._cache[method]

    @classmethod
    def _load(cls, method: ModelMethod) -> Classifier:
        if method == ModelMethod.ZERO_SHOT:
            from app.ai.clip_zero_shot import ZeroShotClassifier
            return ZeroShotClassifier()

        elif method == ModelMethod.TIP_ADAPTER_F:
            from app.ai.tip_adapter_f import TipAdapterFClassifier
            return TipAdapterFClassifier()

        elif method == ModelMethod.CLIP_ADAPTER:
            from app.ai.clip_adapter import CLIPAdapterClassifier
            return CLIPAdapterClassifier()

        elif method == ModelMethod.COOP:
            from app.ai.coop import CoOpClassifier
            return CoOpClassifier()

        raise ValueError(f"Unknown model method: {method}")

    @classmethod
    def list_models(cls) -> list:
        """Return metadata list for all available models (for /api/v1/models endpoint)."""
        return [
            {"id": method.value, **info}
            for method, info in MODEL_DISPLAY_INFO.items()
        ]
