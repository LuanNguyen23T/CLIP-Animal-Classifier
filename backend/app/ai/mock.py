import time
import random
from typing import Union
from PIL.Image import Image as PILImage
from app.ai.base import Classifier, ClassifierResult, PredictionResultItem

class MockClassifier(Classifier):
    def __init__(self, delay_seconds: int = 2):
        self.delay_seconds = delay_seconds

    def predict(self, image: Union[str, bytes, PILImage]) -> ClassifierResult:
        """Mock prediction that sleeps for a few seconds and returns random results."""
        time.sleep(self.delay_seconds)
        
        # Hardcode some fake results for testing UI
        top_k = [
            PredictionResultItem(breed_id=1, breed_name="Abyssinian", species="Cat", score=0.85),
            PredictionResultItem(breed_id=2, breed_name="Bengal", species="Cat", score=0.10),
            PredictionResultItem(breed_id=15, breed_name="pug", species="Dog", score=0.03),
        ]
        
        # Shuffle slightly to feel dynamic
        random.shuffle(top_k)
        # Sort by score descending
        top_k = sorted(top_k, key=lambda x: x.score, reverse=True)

        return ClassifierResult(
            predicted_breed_id=top_k[0].breed_id,
            score=top_k[0].score,
            top_k_results=top_k,
            model_version="mock-classifier-v1"
        )
