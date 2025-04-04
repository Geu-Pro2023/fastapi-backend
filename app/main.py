import os
import zipfile
import numpy as np
import cv2
import json
import sys
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import uuid
import tensorflow as tf
from contextlib import asynccontextmanager
import atexit
from pathlib import Path

# ======================
# CONFIGURATION
# ======================
IS_PRODUCTION = os.getenv("RENDER", False)  # True on Render.com

# Path Configuration
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
LATEST_RESULTS_FILE = BASE_DIR / "latest_results.json"

# Production vs Development paths
if IS_PRODUCTION:
    TEMP_DIR = Path("/tmp/wildguard_uploads")  # More reliable in cloud
    os.makedirs(TEMP_DIR, exist_ok=True)
else:
    TEMP_DIR = BASE_DIR / "temp_uploads"

# Ensure directories exist
STATIC_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Constants
IMG_SIZE = (150, 150)
MODEL_PATH = MODEL_DIR / "retrained_model2_l2_adam.h5"

# ======================
# CLEANUP REGISTRATION
# ======================
def cleanup_temp_files():
    """Clean up temporary files on application exit"""
    for temp_file in TEMP_DIR.glob("*"):
        if temp_file.is_dir():
            shutil.rmtree(temp_file, ignore_errors=True)
        else:
            temp_file.unlink(missing_ok=True)

atexit.register(cleanup_temp_files)

# ======================
# FASTAPI SETUP
# ======================
class TerminalProgress(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Epoch {epoch + 1}/{self.params['epochs']} - "
              f"loss: {logs.get('loss'):.4f} - "
              f"accuracy: {logs.get('accuracy'):.4f} - "
              f"val_loss: {logs.get('val_loss'):.4f} - "
              f"val_accuracy: {logs.get('val_accuracy'):.4f}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Configure TensorFlow
    tf.config.run_functions_eagerly(True)
    
    # Load model with validation
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    try:
        app.state.model = load_model(MODEL_PATH)
        app.state.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        yield
    finally:
        if hasattr(app.state, 'model'):
            del app.state.model
        plt.close('all')

app = FastAPI(
    title="WildGuard API",
    description="API for Endangered Animal Classification",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api-docs",
    redoc_url=None
)

# CORS Middleware (Production-ready)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Frontend dev
        "https://your-frontend.com"  # Production frontend
    ] if IS_PRODUCTION else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    max_age=600
)

# Static files mount
app.mount("/api/static", StaticFiles(directory=STATIC_DIR), name="static")

# ======================
# HELPER FUNCTIONS
# ======================
def save_plot_to_file(fig, filename_prefix):
    """Save matplotlib figure to static directory"""
    filename = f"{filename_prefix}_{uuid.uuid4().hex}.png"
    filepath = STATIC_DIR / filename
    fig.savefig(filepath, bbox_inches='tight', dpi=100)
    plt.close(fig)
    return filename

def preprocess_image(image):
    """Resize and normalize image for model prediction"""
    image = cv2.resize(image, IMG_SIZE)
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# ======================
# API ENDPOINTS
# ======================
@app.get("/")
async def root():
    return {
        "message": "WildGuard API is running",
        "docs": "/api-docs",
        "health_check": "/health",
        "environment": "production" if IS_PRODUCTION else "development"
    }

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": MODEL_PATH.exists(),
        "disk_space_gb": f"{shutil.disk_usage('/').free / (1024**3):.1f}",
        "python_version": sys.version.split()[0],
        "temp_dir": str(TEMP_DIR),
        "is_production": IS_PRODUCTION
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Endpoint for image predictions"""
    temp_dir = TEMP_DIR / f"predict_{uuid.uuid4().hex}"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Save uploaded file
        file_path = temp_dir / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process image
        img = cv2.imread(str(file_path))
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
            
        processed_img = preprocess_image(img)
        confidence_score = app.state.model.predict(processed_img, verbose=0)[0][0]
        confidence_percentage = round(confidence_score * 100, 2)
        predicted_class = "Endangered" if confidence_score < 0.5 else "Non-Endangered"
        
        # Save result image
        img_filename = f"prediction_{uuid.uuid4().hex}.png"
        cv2.imwrite(str(STATIC_DIR / img_filename), img)
        
        return {
            "status": "success",
            "prediction": {
                "class": predicted_class,
                "confidence": confidence_percentage,
                "image_url": f"/api/static/{img_filename}"
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.post("/retrain")
async def retrain(
    file: UploadFile = File(...),
    batch_size: int = Form(32),
    epochs: int = Form(10)
):
    """Endpoint for model retraining"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_dir = TEMP_DIR / f"train_{timestamp}"
    train_dir.mkdir(exist_ok=True)
    
    try:
        # Save and extract dataset
        zip_path = train_dir / file.filename
        with zip_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(train_dir / "dataset")
        
        # Load and preprocess images
        def load_images(directory):
            images = []
            for img_path in directory.glob("*"):
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.resize(img, IMG_SIZE)
                    images.append(img / 255.0)
            return np.array(images)
        
        endangered_dir = train_dir / "dataset" / "Endangered"
        non_endangered_dir = train_dir / "dataset" / "Non-Endangered"
        
        X_endangered = load_images(endangered_dir)
        X_non_endangered = load_images(non_endangered_dir)
        
        if len(X_endangered) == 0 or len(X_non_endangered) == 0:
            raise HTTPException(
                status_code=400,
                detail="Dataset must contain both 'Endangered' and 'Non-Endangered' folders with images"
            )

        # Prepare data
        y_endangered = np.zeros(len(X_endangered))
        y_non_endangered = np.ones(len(X_non_endangered))
        X = np.concatenate([X_endangered, X_non_endangered])
        y = np.concatenate([y_endangered, y_non_endangered])
        
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        history = app.state.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[TerminalProgress()]
        )
        
        # Generate metrics
        y_pred = app.state.model.predict(X_val, verbose=0)
        y_pred_classes = (y_pred > 0.5).astype(int)
        
        # Visualization 1: Class Balance
        fig1 = plt.figure(figsize=(6, 4))
        sns.barplot(x=['Endangered', 'Non-Endangered'],
                   y=[len(X_endangered), len(X_non_endangered)],
                   palette="viridis")
        plt.title('Class Distribution')
        plt.ylabel('Number of Images')
        balance_filename = save_plot_to_file(fig1, "class_balance")
        
        # Visualization 2: Confusion Matrix
        fig2 = plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_val, y_pred_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Endangered', 'Non-Endangered'],
                   yticklabels=['Endangered', 'Non-Endangered'])
        plt.title('Confusion Matrix')
        cm_filename = save_plot_to_file(fig2, "confusion_matrix")
        
        # Visualization 3: ROC Curve
        fpr, tpr, _ = roc_curve(y_val, y_pred)
        roc_auc = auc(fpr, tpr)
        fig3 = plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', label=f'ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        roc_filename = save_plot_to_file(fig3, "roc_curve")
        
        # Save results
        retrain_results = {
            "status": "success",
            "timestamp": timestamp,
            "training_parameters": {
                "batch_size": batch_size,
                "epochs": epochs
            },
            "visualizations": {
                "class_balance": {"url": f"/api/static/{balance_filename}"},
                "confusion_matrix": {"url": f"/api/static/{cm_filename}"},
                "roc_curve": {"url": f"/api/static/{roc_filename}"}
            },
            "metrics": {
                "accuracy": history.history['accuracy'][-1],
                "val_accuracy": history.history['val_accuracy'][-1],
                "loss": history.history['loss'][-1],
                "val_loss": history.history['val_loss'][-1],
                "class_counts": {
                    "endangered": len(X_endangered),
                    "non_endangered": len(X_non_endangered)
                },
                "roc_auc": roc_auc
            }
        }

        with LATEST_RESULTS_FILE.open("w") as f:
            json.dump(retrain_results, f)

        return retrain_results
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )
    finally:
        plt.close('all')
        shutil.rmtree(train_dir, ignore_errors=True)

@app.get("/latest-results")
async def get_latest_results():
    """Get results from the most recent training session"""
    if not LATEST_RESULTS_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="No training results available. Please run /retrain first."
        )
    
    with LATEST_RESULTS_FILE.open("r") as f:
        return json.load(f)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)