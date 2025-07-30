import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import io
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import base64
from PIL import Image
import os
import shutil
from typing import Dict, Any, Optional, List


app = FastAPI(title="Breast Cancer Classification API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

UNET_MODEL_PATH = os.path.join(MODELS_DIR, "saved_unet_model.keras")
CLASS_MODEL_PATH = os.path.join(MODELS_DIR, "6_best_model.h5")

IMG_SIZE = (256, 256)
CLASS_IMG_SIZE = (224, 224)

MODEL_LOAD_STATUS = {
    "segmentation": {"loaded": False, "error": None},
    "classification": {"loaded": False, "error": None}
}

def dice_score(y_true, y_pred):
    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )

def focal_loss(gamma=2., alpha=0.5):
    def loss_fn(y_true, y_pred):
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return -tf.reduce_mean(
            alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt + 1e-7)
        )
    return loss_fn

class SegmentationModel:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        try:
            if not os.path.exists(UNET_MODEL_PATH):
                MODEL_LOAD_STATUS["segmentation"]["error"] = f"File not found: {UNET_MODEL_PATH}"
                print(f"Segmentation model not found at {UNET_MODEL_PATH}")
                return
                
            self.model = load_model(
                UNET_MODEL_PATH,
                compile=False,
                custom_objects={'dice_score': dice_score}
            )
            MODEL_LOAD_STATUS["segmentation"]["loaded"] = True
            print("Segmentation model loaded successfully")
        except Exception as e:
            MODEL_LOAD_STATUS["segmentation"]["error"] = str(e)
            print(f"Error loading segmentation model: {e}")
    
    def predict(self, image_array):
        """Generate segmentation mask for input image"""
        if self.model is None:
            raise RuntimeError("Segmentation model not loaded")
            
        # Ensure image is normalized 0-1
        if image_array.max() > 1.0:
            image_array = image_array.astype('float32') / 255.0
            
        # Predict mask
        mask_prob = self.model.predict(image_array[np.newaxis, ...], verbose=0)
        mask_bin = (mask_prob > 0.5).astype('float32')[0]
        
        return mask_bin

class ClassificationModel:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        try:
            if not os.path.exists(CLASS_MODEL_PATH):
                MODEL_LOAD_STATUS["classification"]["error"] = f"File not found: {CLASS_MODEL_PATH}"
                print(f"Classification model not found at {CLASS_MODEL_PATH}")
                return
                
            self.model = load_model(CLASS_MODEL_PATH, compile=False)
            self.model.compile(
                optimizer='adam',
                loss=focal_loss(),
                metrics=['accuracy']
            )
            MODEL_LOAD_STATUS["classification"]["loaded"] = True
            print("Classification model loaded successfully")
        except Exception as e:
            MODEL_LOAD_STATUS["classification"]["error"] = str(e)
            print(f"Error loading classification model: {e}")
    
    def predict(self, segmented_image, image_name=None):
        """Classify the segmented image"""
        if self.model is None:
            raise RuntimeError("Classification model not loaded")
        preprocessed = preprocess_input(segmented_image * 255.0)
        
        prob = self.model.predict(preprocessed[np.newaxis, ...], verbose=0).ravel()[0]
        
        label = int(prob >= 0.55)
        diagnosis = "Malignant" if label == 1 else "Benign"
            
        return {
            "diagnosis": diagnosis,
            "confidence": float(prob),
            "class_idx": label
        }
        
        

# Initialize models
segmentation_model = SegmentationModel()
classification_model = ClassificationModel()


def create_overlay_image(original_img, mask):
    """Create a colored overlay of the segmentation mask on the original image"""
    overlay = original_img.copy()
    
    colored_mask = np.zeros_like(original_img)
    
    if len(mask.shape) == 2:
        colored_mask[:, :, 0] = mask * 255  
    elif len(mask.shape) == 3 and mask.shape[2] == 1:
        colored_mask[:, :, 0] = mask[:, :, 0] * 255 
    elif len(mask.shape) == 3 and mask.shape[2] == 3:
        colored_mask[:, :, 0] = mask[:, :, 0] * 255
    
    alpha = 0.5
    overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)
    
    return overlay

def encode_image_to_base64(image_array):
    """Convert numpy image array to base64 string"""
    # Ensure the image is in BGR format for OpenCV
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # Convert to uint8 if floating point
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        success, encoded_image = cv2.imencode('.png', image_array)
        if not success:
            raise Exception("Could not encode image to PNG format")
        base64_string = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
        return base64_string
    else:
        raise ValueError(f"Unexpected image shape: {image_array.shape}")

# API Endpoints
@app.get("/")
async def root():
    """API health check endpoint."""
    return {
        "status": "ok", 
        "message": "Breast Cancer Classification API is running",
        "models_loaded": {
            "segmentation": MODEL_LOAD_STATUS["segmentation"]["loaded"],
            "classification": MODEL_LOAD_STATUS["classification"]["loaded"]
        }
    }

@app.get("/status")
async def status():
    """Detailed model loading status"""
    return {
        "segmentation_model": {
            "path": UNET_MODEL_PATH,
            "exists": os.path.exists(UNET_MODEL_PATH),
            "loaded": MODEL_LOAD_STATUS["segmentation"]["loaded"],
            "error": MODEL_LOAD_STATUS["segmentation"]["error"]
        },
        "classification_model": {
            "path": CLASS_MODEL_PATH,
            "exists": os.path.exists(CLASS_MODEL_PATH),
            "loaded": MODEL_LOAD_STATUS["classification"]["loaded"],
            "error": MODEL_LOAD_STATUS["classification"]["error"]
        }
    }

@app.post("/upload-model")
async def upload_model(
    model_type: str = Form(...),
    model_file: UploadFile = File(...)
):
    """Upload a model file"""
    if model_type not in ["segmentation", "classification"]:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    try:
        # Determine destination path
        if model_type == "segmentation":
            dest_path = UNET_MODEL_PATH
        else:
            dest_path = CLASS_MODEL_PATH
        
        # Save the file
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(model_file.file, buffer)

        if model_type == "segmentation":
            segmentation_model.load_model()
        else:
            classification_model.load_model()
        
        return {"message": f"{model_type} model uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading model: {str(e)}")

@app.post("/reload-models")
async def reload_models():
    """Force reload the models"""
    segmentation_model.load_model()
    classification_model.load_model()
    return {
        "segmentation": MODEL_LOAD_STATUS["segmentation"],
        "classification": MODEL_LOAD_STATUS["classification"]
    }

@app.post("/analyze")
async def analyze_image(image: UploadFile = File(...), image_name: str = Form(None)):
    """Analyze a mammogram image for breast cancer detection."""
    # Check if models are loaded
    if not MODEL_LOAD_STATUS["segmentation"]["loaded"]:
        raise HTTPException(
            status_code=500, 
            detail=f"Segmentation model not loaded: {MODEL_LOAD_STATUS['segmentation']['error']}"
        )
    
    if not MODEL_LOAD_STATUS["classification"]["loaded"]:
        raise HTTPException(
            status_code=500, 
            detail=f"Classification model not loaded: {MODEL_LOAD_STATUS['classification']['error']}"
        )
    
    try:
        content = await image.read()
        
        if not image_name:
            image_name = image.filename
            
        print(f"Processing image: {image_name}")
        
        pil_image = Image.open(io.BytesIO(content))
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        resized_image = cv2.resize(opencv_image, IMG_SIZE)
        resized_array = resized_image.astype('float32') / 255.0
        
        segmentation_mask = segmentation_model.predict(resized_array)
        
        segmented_img = resized_array * segmentation_mask
        display_img = (resized_image).astype(np.uint8)
        segmentation_overlay = create_overlay_image(display_img, segmentation_mask)
        
        seg_img_cls = cv2.resize(segmented_img, CLASS_IMG_SIZE)
        classification_result = classification_model.predict(seg_img_cls, image_name)
        segmentation_base64 = encode_image_to_base64(segmentation_overlay)
        
        # Return results
        return JSONResponse(content={
            "diagnosis": classification_result["diagnosis"],
            "confidence": float(classification_result["confidence"]),
            "class_idx": classification_result["class_idx"],
            "segmentation_image": segmentation_base64
        })
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)