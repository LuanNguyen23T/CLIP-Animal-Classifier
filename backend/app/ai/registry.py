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
    TIP_ADAPTER  = "tip-adapter"
    TIP_ADAPTER_F = "tip-adapter-f"
    CLIP_ADAPTER = "clip-adapter"
    COOP         = "coop"
    RESNET50_PROTOTYPE = "resnet50-prototype"
    RESNET50_BASELINE  = "resnet50-baseline"
    RESNET50_LINEAR    = "resnet50-linear"
    RESNET50_ADAPTER   = "resnet50-adapter"
    RESNET50_COSINE    = "resnet50-cosine"


# Human-readable display names for the UI
MODEL_DISPLAY_INFO: Dict[str, dict] = {
    ModelMethod.ZERO_SHOT: {
        "name":        "Zero-Shot",
        "description": "Baseline CLIP using manually crafted text prompts.",
        "accuracy":    87.76,
        "latency_ms":  0.85,
        "params":      0,
        "tag":         "Fastest",
        "backbone":    "clip",
    },
    ModelMethod.TIP_ADAPTER: {
        "name":        "Tip-Adapter (Free)",
        "description": "Training-free cache model using few-shot support set.",
        "accuracy":    87.90,
        "latency_ms":  0.92,
        "params":      0,
        "tag":         "Training-Free",
        "backbone":    "clip",
    },
    ModelMethod.TIP_ADAPTER_F: {
        "name":        "Tip-Adapter-F",
        "description": "Fine-tuned cache model keys for optimal few-shot performance.",
        "accuracy":    89.56,
        "latency_ms":  0.92,
        "params":      1884160,
        "tag":         "Few-Shot Tuned",
        "backbone":    "clip",
    },
    ModelMethod.CLIP_ADAPTER: {
        "name":        "CLIP-Adapter",
        "description": "Residual feature adapter with bottleneck MLP.",
        "accuracy":    90.27,
        "latency_ms":  0.89,
        "params":      66112,
        "tag":         "Feature Tuned",
        "backbone":    "clip",
    },
    ModelMethod.COOP: {
        "name":        "CoOp (Prompt Tuning)",
        "description": "Learnable context vectors injected into CLIP text prompts.",
        "accuracy":    91.61,
        "latency_ms":  0.95,
        "params":      2048,
        "tag":         "Best Accuracy",
        "backbone":    "clip",
    },
    ModelMethod.RESNET50_PROTOTYPE: {
        "name":        "ResNet-50 Prototype",
        "description": "Nearest centroid classification using frozen ResNet features.",
        "accuracy":    91.69,
        "latency_ms":  18.50,
        "params":      0,
        "tag":         "Training-Free",
        "backbone":    "resnet50",
    },
    ModelMethod.RESNET50_BASELINE: {
        "name":        "ResNet-50 Baseline",
        "description": "Standard linear probe on raw images without TTA.",
        "accuracy":    91.80,
        "latency_ms":  4.50,
        "params":      75813,
        "tag":         "Standard",
        "backbone":    "resnet50",
    },
    ModelMethod.RESNET50_LINEAR: {
        "name":        "ResNet-50 Linear Probe",
        "description": "Standard linear layer trained on frozen features.",
        "accuracy":    92.40,
        "latency_ms":  18.60,
        "params":      75813,
        "tag":         "Standard Probe",
        "backbone":    "resnet50",
    },
    ModelMethod.RESNET50_ADAPTER: {
        "name":        "ResNet-50 Residual Adapter",
        "description": "Residual MLP with bottleneck architecture.",
        "accuracy":    92.42,
        "latency_ms":  19.10,
        "params":      1050213,
        "tag":         "Deep Probe",
        "backbone":    "resnet50",
    },
    ModelMethod.RESNET50_COSINE: {
        "name":        "ResNet-50 Cosine Head",
        "description": "Cosine distance classifier with 5-view TTA.",
        "accuracy":    93.08,
        "latency_ms":  20.45,
        "params":      75815,
        "tag":         "Best CNN",
        "backbone":    "resnet50",
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

        elif method == ModelMethod.TIP_ADAPTER:
            from app.ai.tip_adapter import TipAdapterClassifier
            return TipAdapterClassifier()

        elif method == ModelMethod.TIP_ADAPTER_F:
            from app.ai.tip_adapter_f import TipAdapterFClassifier
            return TipAdapterFClassifier()

        elif method == ModelMethod.CLIP_ADAPTER:
            from app.ai.clip_adapter import CLIPAdapterClassifier
            return CLIPAdapterClassifier()

        elif method == ModelMethod.COOP:
            from app.ai.coop import CoOpClassifier
            return CoOpClassifier()

        elif method == ModelMethod.RESNET50_PROTOTYPE:
            from app.ai.resnet50_prototype import ResNet50PrototypeClassifier
            return ResNet50PrototypeClassifier()

        elif method == ModelMethod.RESNET50_BASELINE:
            from app.ai.resnet50_baseline import ResNet50BaselineClassifier
            return ResNet50BaselineClassifier()
            
        elif method == ModelMethod.RESNET50_LINEAR:
            from app.ai.resnet50_linear import ResNet50LinearClassifier
            return ResNet50LinearClassifier()
            
        elif method == ModelMethod.RESNET50_ADAPTER:
            from app.ai.resnet50_adapter import ResNet50AdapterClassifier
            return ResNet50AdapterClassifier()

        elif method == ModelMethod.RESNET50_COSINE:
            from app.ai.resnet50_cosine import ResNet50CosineClassifier
            return ResNet50CosineClassifier()

        raise ValueError(f"Unknown model method: {method}")

    @classmethod
    def list_models(cls) -> list:
        """Return metadata list for all available models (for /api/v1/models endpoint)."""
        return [
            {"id": method.value, **info}
            for method, info in MODEL_DISPLAY_INFO.items()
        ]
