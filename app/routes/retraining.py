from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
import zipfile
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from ...models.model_utils import load_dataset, retrain_model, evaluate_model

router = APIRouter()

@router.post("/retrain")
async def retrain(file: UploadFile = File(...)):
    try:
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_dir = os.path.join("temp_uploads", f"train_{timestamp}")
        os.makedirs(train_dir, exist_ok=True)
        
        # Save uploaded zip file
        zip_path = os.path.join(train_dir, file.filename)
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract zip file
        extract_path = os.path.join(train_dir, "dataset")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # Load and preprocess data
        X, y = load_dataset(extract_path)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Retrain model
        history = retrain_model(X_train, y_train, X_val, y_val)
        
        # Evaluate model
        metrics = evaluate_model(X_val, y_val, history)
        
        return {
            "success": True,
            "message": "Model retrained successfully",
            "timestamp": timestamp,
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(train_dir, ignore_errors=True)