from typing import List, Optional
from uuid import UUID, uuid4
from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload
from app.core.database import get_db
from app.schemas.prediction import (
    PredictionCreateResponse,
    PredictionHistoryResponse,
    PredictionResponse,
)
from app.models.prediction import PredictionStatus
from app.models.prediction import Prediction
from app.models.breed import Breed
from app.repositories import prediction as prediction_repo
from app.services.prediction import run_inference_task
from app.storage.cloudinary_storage import upload_image_to_cloudinary
# from app.storage.local import LocalStorage
from app.ai.registry import ClassifierRegistry, MODEL_DISPLAY_INFO, ModelMethod

router = APIRouter()

@router.post("/predictions", response_model=PredictionCreateResponse)
async def create_prediction(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    method: ModelMethod = ModelMethod.ZERO_SHOT,
    db: Session = Depends(get_db)
):
    """Upload an image and start background classification."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
        
    # Read file content and upload to Cloudinary
    file_content = await file.read()
    image_url = upload_image_to_cloudinary(file_content, file.filename)
    
    # Create DB record (status: UPLOADED)
    prediction = prediction_repo.create_prediction(db, image_key=image_url, method=method)
    
    # Start background task
    # We pass a new db session for the background task to avoid session conflicts?
    # For now in Phase 1, FastAPI depends(get_db) will close the session after response.
    # We should use a fresh session in the background task.
    background_tasks.add_task(
        background_task_runner, 
        prediction.id,
        method
    )
    
    return prediction

@router.post("/predictions/batch", response_model=List[PredictionCreateResponse])
async def create_prediction_batch(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    """Upload an image and start background classification for ALL models."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
        
    # Read file content and upload to Cloudinary once
    file_content = await file.read()
    image_url = upload_image_to_cloudinary(file_content, file.filename)
    
    predictions = []
    batch_id = uuid4()
    # Create DB record and start background task for each model method
    for method in ModelMethod:
        prediction = prediction_repo.create_prediction(
            db,
            image_key=image_url,
            method=method,
            batch_id=batch_id,
        )
        background_tasks.add_task(background_task_runner, prediction.id, method)
        predictions.append(prediction)
        
    return predictions

def background_task_runner(prediction_id: UUID, method: str):
    # Create a fresh DB session for the background thread
    from app.core.database import SessionLocal
    db = SessionLocal()
    try:
        classifier = ClassifierRegistry.get(ModelMethod(method))
        run_inference_task(db, prediction_id, classifier)
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
    method: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get history of predictions."""
    return prediction_repo.get_predictions(db, status=status, breed_id=breed_id, method=method, skip=skip, limit=limit)


@router.get("/prediction-history", response_model=PredictionHistoryResponse)
def get_prediction_history(
    status: Optional[PredictionStatus] = None,
    backbone: Optional[str] = None,
    method: Optional[str] = None,
    page: int = 1,
    limit: int = 12,
    db: Session = Depends(get_db),
):
    """Return history paginated by classification request, not model run."""
    if page < 1:
        raise HTTPException(status_code=422, detail="page must be at least 1")
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100")

    methods = None
    if backbone:
        methods = [
            model_method.value
            for model_method, info in MODEL_DISPLAY_INFO.items()
            if info["backbone"] == backbone
        ]
        if not methods:
            raise HTTPException(status_code=422, detail="Unknown backbone")

    group_key = func.coalesce(Prediction.batch_id, Prediction.id)
    matching_filters = []
    if status:
        matching_filters.append(Prediction.status == status)
    if method:
        matching_filters.append(Prediction.method == method)
    elif methods:
        matching_filters.append(Prediction.method.in_(methods))

    group_query = (
        db.query(
            group_key.label("group_id"),
            func.max(Prediction.created_at).label("created_at"),
        )
        .filter(*matching_filters)
        .group_by(group_key)
    )
    total = group_query.count()
    group_rows = (
        group_query
        .order_by(func.max(Prediction.created_at).desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )
    group_ids = [row.group_id for row in group_rows]

    predictions = []
    if group_ids:
        predictions = (
            db.query(Prediction)
            .options(joinedload(Prediction.predicted_breed).joinedload(Breed.species))
            .filter(group_key.in_(group_ids))
            .order_by(Prediction.created_at.desc())
            .all()
        )

    grouped = {
        str(row.group_id): {
            "group_id": str(row.group_id),
            "batch_id": None,
            "image_key": "",
            "created_at": row.created_at,
            "predictions": [],
        }
        for row in group_rows
    }
    for prediction in predictions:
        group = grouped[str(prediction.batch_id or prediction.id)]
        group["batch_id"] = prediction.batch_id
        group["image_key"] = prediction.image_key
        group["predictions"].append(prediction)

    groups = list(grouped.values())
    return {
        "items": groups,
        "total": total,
        "page": page,
        "limit": limit,
    }
