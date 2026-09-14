import traceback
from uuid import UUID
from sqlalchemy.orm import Session
from app.repositories import prediction as prediction_repo
from app.models.prediction import PredictionStatus
from app.storage.base import Storage
from app.ai.base import Classifier

def run_inference_task(
    db: Session, 
    prediction_id: UUID, 
    storage: Storage, 
    classifier: Classifier
):
    """
    Background task to run inference.
    """
    db_obj = prediction_repo.get_prediction(db, prediction_id)
    if not db_obj:
        return
        
    try:
        # Mark as processing
        prediction_repo.update_prediction_status(db, db_obj, PredictionStatus.PROCESSING)
        
        # Get absolute path from storage
        image_path = storage.get_path(db_obj.image_key)
        
        # Run classifier
        result = classifier.predict(image_path)
        
        # Convert top_k to serializable dicts
        top_k_dicts = [item.model_dump() for item in result.top_k_results]
        
        # Mark as completed
        prediction_repo.update_prediction_status(
            db, 
            db_obj, 
            status=PredictionStatus.COMPLETED,
            predicted_breed_id=result.predicted_breed_id,
            score=result.score,
            top_k_results=top_k_dicts,
            model_version=result.model_version
        )
        
    except Exception as e:
        error_msg = traceback.format_exc()
        # Mark as failed
        prediction_repo.update_prediction_status(
            db, 
            db_obj, 
            status=PredictionStatus.FAILED,
            error_message=str(e) + "\n" + error_msg
        )
