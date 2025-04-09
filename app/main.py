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
IS_PRODUCTION = os.getenv("RENDER", "false").lower() == "true"

# Path Configuration
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
LATEST_RESULTS_FILE = BASE_DIR / "latest_results.json"

# Production vs Development paths
if IS_PRODUCTION:
    TEMP_DIR = Path("/tmp/wildguard_uploads")
    STATIC_DIR = Path("/opt/render/project/src/app/static")
    # Ensure production directories exist
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)
else:
    TEMP_DIR = BASE_DIR / "temp_uploads"
    STATIC_DIR = BASE_DIR / "static"

# Constants
IMG_SIZE = (150, 150)
MODEL_PATH = MODEL_DIR / "retrained_model2_l2_adam.h5"

# ======================
# MODEL VERIFICATION
# ======================
def verify_model():
    """Verify model exists and is accessible"""
    if not MODEL_PATH.exists():
        alt_path = Path("/opt/render/project/src/app/models/retrained_model2_l2_adam.h5")
        if alt_path.exists():
            return alt_path
        raise FileNotFoundError(
            f"Model not found at either {MODEL_PATH} or {alt_path}\n"
            f"Current working directory: {Path.cwd()}\n"
            f"Directory contents: {list(Path.cwd().rglob('*'))}"
        )
    return MODEL_PATH

# ======================
# CLEANUP REGISTRATION
# ======================
def cleanup_temp_files():
    """Clean up temporary files on application exit"""
    for temp_file in TEMP_DIR.glob("*"):
        try:
            if temp_file.is_dir():
                shutil.rmtree(temp_file, ignore_errors=True)
            else:
                temp_file.unlink(missing_ok=True)
        except Exception as e:
            print(f"Failed to clean {temp_file}: {str(e)}")

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
    # Configure TensorFlow for production
    tf.config.run_functions_eagerly(True)
    if IS_PRODUCTION:
        tf.config.set_visible_devices([], 'GPU')
        print("Configured TensorFlow to use CPU only in production")
    
    # Load model with validation
    actual_model_path = verify_model()
    print(f"Loading model from: {actual_model_path}")
    
    try:
        app.state.model = load_model(actual_model_path)
        app.state.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print("Model loaded successfully")
        yield
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")
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

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-frontend.com"
    ] if IS_PRODUCTION else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
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
# API ENDPOINTS (Keep your existing endpoints exactly as they are)
# ======================
# [Keep all your existing endpoints unchanged - root, health, predict, retrain, latest-results]

# ======================
# DEBUG ENDPOINTS (For production troubleshooting)
# ======================
@app.get("/debug/paths")
async def debug_paths():
    """Debug endpoint to verify file paths"""
    return {
        "is_production": IS_PRODUCTION,
        "base_dir": str(BASE_DIR),
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "static_dir": str(STATIC_DIR),
        "static_dir_exists": STATIC_DIR.exists(),
        "temp_dir": str(TEMP_DIR),
        "temp_dir_exists": TEMP_DIR.exists(),
        "current_working_dir": str(Path.cwd()),
        "disk_usage": shutil.disk_usage("/")._asdict()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
