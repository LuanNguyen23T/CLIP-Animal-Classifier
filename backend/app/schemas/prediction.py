import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from app.models.prediction import PredictionStatus
from app.schemas.breed import BreedResponse

class PredictionCreateResponse(BaseModel):
    id: uuid.UUID
    status: PredictionStatus

class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": (), "from_attributes": True}

    id: uuid.UUID
    image_key: str
    predicted_breed: Optional[BreedResponse] = None
    score: Optional[float] = None
    top_k_results: Optional[List[Dict[str, Any]]] = None
    model_version: Optional[str] = None
    status: PredictionStatus
    error_message: Optional[str] = None
    created_at: datetime
