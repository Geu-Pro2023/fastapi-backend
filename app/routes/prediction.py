from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import os
import cv2
import numpy as np
from datetime import datetime
from ...models.model_utils import preprocess_image, predict_image

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pred_dir = os.path.join("temp_uploads", f"predict_{timestamp}")
        os.makedirs(pred_dir, exist_ok=True)
        
        # Save uploaded file
        file_path = os.path.join(pred_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Preprocess and predict
        img = cv2.imread(file_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
            
        processed_img = preprocess_image(img)
        confidence_score, predicted_class = predict_image(processed_img)
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence_score * 100,
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(pred_dir, ignore_errors=True)