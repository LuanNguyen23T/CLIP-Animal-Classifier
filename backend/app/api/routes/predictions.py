from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.prediction import PredictionResponse, PredictionCreateResponse
from app.models.prediction import PredictionStatus
from app.repositories import prediction as prediction_repo
from app.services.prediction import run_inference_task
from app.storage.local import LocalStorage
# from app.ai.mock import MockClassifier  # Phase 1-3: MockClassifier (rollback nếu cần)
from app.ai.clip import CLIPZeroShotClassifier

router = APIRouter()

# Phase 4: CLIP Zero-Shot Classifier (tạm thời, chờ R&D chọn model tốt nhất)
# Để rollback về Mock: thay dòng dưới bằng MockClassifier(delay_seconds=2)
storage = LocalStorage()
classifier = CLIPZeroShotClassifier(use_tta=False)

@router.post("/predictions", response_model=PredictionCreateResponse)
def create_prediction(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    """Upload an image and start background classification."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
        
    # Save file
    image_key = storage.save(file.file, file.filename)
    
    # Create DB record (status: UPLOADED)
    prediction = prediction_repo.create_prediction(db, image_key=image_key)
    
    # Start background task
    # We pass a new db session for the background task to avoid session conflicts?
    # For now in Phase 1, FastAPI depends(get_db) will close the session after response.
    # We should use a fresh session in the background task.
    background_tasks.add_task(
        background_task_runner, 
        prediction.id
    )
    
    return prediction

def background_task_runner(prediction_id: UUID):
    # Create a fresh DB session for the background thread
    from app.core.database import SessionLocal
    db = SessionLocal()
    try:
        run_inference_task(db, prediction_id, storage, classifier)
    finally:
        db.close()

@router.get("/predictions/{prediction_id}", response_model=PredictionResponse)
def get_prediction(prediction_id: UUID, db: Session = Depends(get_db)):
    """Get status and result of a prediction."""
    prediction = prediction_repo.get_prediction(db, prediction_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return prediction

@router.get("/predictions", response_model=List[PredictionResponse])
def get_predictions(
    status: Optional[PredictionStatus] = None,
    breed_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get history of predictions."""
    return prediction_repo.get_predictions(db, status=status, breed_id=breed_id, skip=skip, limit=limit)
