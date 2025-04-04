import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix

# Constants
IMG_SIZE = (150, 150)
MODEL_PATH = os.path.join("models", "retrained_model2_l2_adam.h5")
model = load_model(MODEL_PATH)

def preprocess_image(img):
    """Preprocess image for prediction"""
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def predict_image(img_array):
    """Make prediction on single image"""
    confidence_score = model.predict(img_array, verbose=0)[0][0]
    predicted_class = "Endangered" if confidence_score < 0.5 else "Non-Endangered"
    return confidence_score, predicted_class

def load_dataset(dataset_dir):
    """Load and preprocess dataset from directory"""
    endangered_dir = os.path.join(dataset_dir, "Endangered")
    non_endangered_dir = os.path.join(dataset_dir, "Non-Endangered")
    
    def load_images(directory):
        images = []
        valid_extensions = ('.jpg', '.jpeg', '.png')
        for img_name in os.listdir(directory):
            if img_name.startswith('.') or not img_name.lower().endswith(valid_extensions):
                continue
            img_path = os.path.join(directory, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, IMG_SIZE)
                img = img / 255.0
                images.append(img)
        return np.array(images)
    
    X_endangered = load_images(endangered_dir)
    X_non_endangered = load_images(non_endangered_dir)
    
    if len(X_endangered) == 0 or len(X_non_endangered) == 0:
        raise ValueError("No valid images found in one or both directories")
    
    y_endangered = np.zeros(len(X_endangered))
    y_non_endangered = np.ones(len(X_non_endangered))
    
    X = np.concatenate([X_endangered, X_non_endangered])
    y = np.concatenate([y_endangered, y_non_endangered])
    
    return X, y

def retrain_model(X_train, y_train, X_val, y_val):
    """Retrain the model with new data"""
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        verbose=0
    )
    model.save(MODEL_PATH)
    return history

def evaluate_model(X_val, y_val, history):
    """Evaluate model and generate metrics"""
    y_pred = model.predict(X_val, verbose=0)
    y_pred_classes = (y_pred > 0.5).astype(int)
    
    # Confusion matrix plot
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_val, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Endangered', 'Non-Endangered'],
                yticklabels=['Endangered', 'Non-Endangered'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    cm_buffer = io.BytesIO()
    plt.savefig(cm_buffer, format='png', bbox_inches='tight')
    plt.close()
    cm_base64 = base64.b64encode(cm_buffer.read()).decode('utf-8')
    
    # Accuracy/Loss plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    metrics_buffer = io.BytesIO()
    plt.savefig(metrics_buffer, format='png', bbox_inches='tight')
    plt.close()
    metrics_base64 = base64.b64encode(metrics_buffer.read()).decode('utf-8')
    
    # Class distribution
    class_counts = {
        "endangered": int(np.sum(y_val == 0)),
        "non_endangered": int(np.sum(y_val == 1))
    }
    
    return {
        "accuracy": history.history['accuracy'][-1],
        "val_accuracy": history.history['val_accuracy'][-1],
        "loss": history.history['loss'][-1],
        "val_loss": history.history['val_loss'][-1],
        "confusion_matrix": cm_base64,
        "metrics_plot": metrics_base64,
        "class_balance": class_counts
    }