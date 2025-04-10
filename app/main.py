import os
import zipfile
import numpy as np
import cv2
import json
import sys
import logging
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    # Create directories if they don't exist
    for dir_path in [TEMP_DIR, STATIC_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
else:
    TEMP_DIR = BASE_DIR / "temp_uploads"
    STATIC_DIR = BASE_DIR / "static"

# Ensure directories exist
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Constants
IMG_SIZE = (150, 150)
MODEL_PATH = MODEL_DIR / "retrained_model2_l2_adam.h5"
VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
MAX_FILE_SIZE_MB = 50  # Maximum allowed file size for uploads

# ======================
# MODEL HANDLING
# ======================
def get_model_path():
    """Get the correct model path for the current environment"""
    if IS_PRODUCTION:
        # Check both possible production paths
        render_path = Path("/opt/render/project/src/app/models/retrained_model2_l2_adam.h5")
        if render_path.exists():
            return render_path
    return MODEL_PATH

# ======================
# FILE HANDLING
# ======================
def save_uploaded_file(upload_file: UploadFile, destination: Path):
    """Save uploaded file with size validation"""
    # Check file size
    file_size = 0
    max_size = MAX_FILE_SIZE_MB * 1024 * 1024  # Convert MB to bytes
    
    with destination.open("wb") as buffer:
        for chunk in upload_file.file:
            file_size += len(chunk)
            if file_size > max_size:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Max size: {MAX_FILE_SIZE_MB}MB"
                )
            buffer.write(chunk)
    
    logger.info(f"Saved file to {destination} (size: {file_size/1024/1024:.2f}MB)")

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
            logger.info(f"Cleaned up temporary file: {temp_file}")
        except Exception as e:
            logger.error(f"Failed to clean {temp_file}: {str(e)}")

atexit.register(cleanup_temp_files)

# ======================
# FASTAPI SETUP
# ======================
class TerminalProgress(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logger.info(f"Epoch {epoch + 1}/{self.params['epochs']} - "
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
        logger.info("Configured TensorFlow to use CPU only in production")
    
    # Get the correct model path
    model_path = get_model_path()
    logger.info(f"Loading model from: {model_path}")
    
    if not model_path.exists():
        error_msg = f"Model file not found at {model_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        app.state.model = load_model(model_path)
        app.state.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        logger.info("Model loaded successfully")
        yield
    except Exception as e:
        error_msg = f"Model loading failed: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
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
    allow_origins=["*"],  # Allow all for testing, restrict in production
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
    logger.info(f"Saved plot to: {filepath}")
    return filename

def preprocess_image(image):
    """Resize and normalize image for model prediction"""
    try:
        image = cv2.resize(image, IMG_SIZE)
        image = image / 255.0
        return np.expand_dims(image, axis=0)
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise ValueError(f"Image processing error: {str(e)}")

def validate_zip_file(zip_path: Path):
    """Validate the structure and content of the training ZIP file"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            
            # Check for required directories
            has_endangered = any('Endangered/' in f for f in file_list)
            has_non_endangered = any('Non-Endangered/' in f for f in file_list)
            
            if not has_endangered or not has_non_endangered:
                raise ValueError("ZIP must contain 'Endangered' and 'Non-Endangered' folders")
                
            # Check for at least one image in each directory
            endangered_images = [f for f in file_list if 'Endangered/' in f and f.lower().endswith(VALID_IMAGE_EXTENSIONS)]
            non_endangered_images = [f for f in file_list if 'Non-Endangered/' in f and f.lower().endswith(VALID_IMAGE_EXTENSIONS)]
            
            if not endangered_images or not non_endangered_images:
                raise ValueError("Each folder must contain at least one valid image file")
                
            return True
    except zipfile.BadZipFile:
        raise ValueError("Invalid ZIP file format")
    except Exception as e:
        raise ValueError(f"ZIP validation error: {str(e)}")

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
    model_path = get_model_path()
    return {
        "status": "healthy",
        "model_loaded": model_path.exists(),
        "model_path": str(model_path),
        "disk_space_gb": f"{shutil.disk_usage('/').free / (1024**3):.1f}",
        "python_version": sys.version.split()[0],
        "temp_dir": str(TEMP_DIR),
        "static_dir": str(STATIC_DIR),
        "is_production": IS_PRODUCTION
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Endpoint for image predictions"""
    temp_dir = TEMP_DIR / f"predict_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created temp directory for prediction: {temp_dir}")
    
    try:
        # Verify file is an image
        if not any(file.filename.lower().endswith(ext) for ext in VALID_IMAGE_EXTENSIONS):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Supported formats: {', '.join(VALID_IMAGE_EXTENSIONS)}"
            )

        # Save uploaded file
        file_path = temp_dir / file.filename
        save_uploaded_file(file, file_path)
        
        # Process image
        img = cv2.imread(str(file_path))
        if img is None:
            raise HTTPException(
                status_code=400,
                detail="Unable to read image file. Please check the file format."
            )
            
        processed_img = preprocess_image(img)
        confidence_score = app.state.model.predict(processed_img, verbose=0)[0][0]
        confidence_percentage = round(confidence_score * 100, 2)
        predicted_class = "Endangered" if confidence_score < 0.5 else "Non-Endangered"
        
        # Save result image
        img_filename = f"prediction_{uuid.uuid4().hex}.png"
        cv2.imwrite(str(STATIC_DIR / img_filename), img)
        logger.info(f"Saved prediction result image: {img_filename}")
        
        return {
            "status": "success",
            "prediction": {
                "class": predicted_class,
                "confidence": confidence_percentage,
                "image_url": f"/api/static/{img_filename}"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info(f"Cleaned up temp directory: {temp_dir}")
        except Exception as e:
            logger.error(f"Failed to clean temp directory: {str(e)}")

@app.post("/retrain")
async def retrain(
    file: UploadFile = File(...),
    batch_size: int = Form(32),
    epochs: int = Form(10)
):
    """Endpoint for model retraining"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_dir = TEMP_DIR / f"train_{timestamp}"
    train_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created training directory: {train_dir}")
    
    try:
        # Verify file is a ZIP file
        if not file.filename.lower().endswith('.zip'):
            raise HTTPException(
                status_code=400,
                detail="Uploaded file must be a .zip file"
            )

        # Save uploaded ZIP file
        zip_path = train_dir / file.filename
        save_uploaded_file(file, zip_path)
        
        # Validate ZIP file structure
        try:
            validate_zip_file(zip_path)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )
        
        # Extract the dataset
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(train_dir / "dataset")
        logger.info("Extracted training dataset")
        
        # Load and preprocess images
        def load_images(directory):
            images = []
            for img_path in directory.glob("*"):
                if img_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img = cv2.resize(img, IMG_SIZE)
                        images.append(img / 255.0)
                    else:
                        logger.warning(f"Could not read image: {img_path}")
            return np.array(images)
        
        endangered_dir = train_dir / "dataset" / "Endangered"
        non_endangered_dir = train_dir / "dataset" / "Non-Endangered"
        
        # Verify extracted directories
        if not endangered_dir.exists() or not non_endangered_dir.exists():
            raise HTTPException(
                status_code=400,
                detail="Extracted folders must be named exactly 'Endangered' and 'Non-Endangered'"
            )

        X_endangered = load_images(endangered_dir)
        X_non_endangered = load_images(non_endangered_dir)
        
        if len(X_endangered) == 0 or len(X_non_endangered) == 0:
            raise HTTPException(
                status_code=400,
                detail="Dataset must contain valid images in both 'Endangered' and 'Non-Endangered' folders"
            )
        
        logger.info(f"Loaded {len(X_endangered)} endangered and {len(X_non_endangered)} non-endangered images")

        # Prepare data
        y_endangered = np.zeros(len(X_endangered))
        y_non_endangered = np.ones(len(X_non_endangered))
        X = np.concatenate([X_endangered, X_non_endangered])
        y = np.concatenate([y_endangered, y_non_endangered])
        
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        logger.info(f"Starting training with {epochs} epochs and batch size {batch_size}")
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
        
        # Prepare results
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

        # Save results to file
        with LATEST_RESULTS_FILE.open("w") as f:
            json.dump(retrain_results, f)
        logger.info("Training completed successfully")
        
        # Return the results directly
        return retrain_results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )
    finally:
        try:
            plt.close('all')
            shutil.rmtree(train_dir, ignore_errors=True)
            logger.info(f"Cleaned up training directory: {train_dir}")
        except Exception as e:
            logger.error(f"Failed to clean training directory: {str(e)}")

@app.get("/latest-results")
async def get_latest_results():
    """Get results from the most recent training session"""
    if not LATEST_RESULTS_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="No training results available. Please run /retrain first."
        )
    
    try:
        with LATEST_RESULTS_FILE.open("r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read latest results: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to load training results"
        )

@app.get("/api/static/{filename}")
async def serve_static(filename: str):
    """Serve static files directly"""
    file_path = STATIC_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
