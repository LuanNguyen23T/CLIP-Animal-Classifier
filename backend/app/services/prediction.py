import traceback
from uuid import UUID
from sqlalchemy.orm import Session
from app.repositories import prediction as prediction_repo
from app.models.prediction import PredictionStatus
from app.ai.base import Classifier

def run_inference_task(
    db: Session, 
    prediction_id: UUID, 
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
        
        import urllib.request
        
        # The image_key is now a Cloudinary URL
        image_url = db_obj.image_key
        
        # Download image into bytes
        req = urllib.request.Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            image_bytes = response.read()
        
        import time
        start_time = time.time()
        
        import io
        from PIL import Image
        from app.services.image import apply_optimal_preprocessing
        
        # Run classifier
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        processed_img = apply_optimal_preprocessing(pil_img)
        result = classifier.predict(processed_img)
        
        processing_time = time.time() - start_time
        
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
            model_version=result.model_version,
            processing_time=processing_time
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
