from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import scipy.io
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import os
import tempfile
import shutil

app = FastAPI(title="Bearing Fault Detection API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and encoder
model = None
encoder = None

def extract_signal(data):
    """Extract vibration signal from .mat file"""
    for key in data:
        if "DE_time" in key:
            return data[key].flatten()
    return None

def segment_signal(signal, window_size=2048, step=1024):
    """Segment signal into windows for CNN input"""
    segments = []
    for start in range(0, len(signal) - window_size, step):
        seg = signal[start:start + window_size]
        # Normalize each segment
        seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-8)
        segments.append(seg)
    return segments

def load_model():
    """Load the trained model and label encoder"""
    global model, encoder
    try:
        model = tf.keras.models.load_model('../bearing_fault_model.h5')
        encoder_classes = np.load('../label_classes.npy')
        encoder = LabelEncoder()
        encoder.classes_ = encoder_classes
        print("Model and encoder loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        encoder = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    return {"message": "Bearing Fault Detection API", "status": "running"}

@app.post("/predict")
async def predict_fault(file: UploadFile = File(...)):
    """
    Predict bearing fault from uploaded .mat file
    """
    if model is None or encoder is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not file.filename.endswith('.mat'):
        raise HTTPException(status_code=400, detail="File must be a .mat file")

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mat') as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        # Load and process the .mat file
        data = scipy.io.loadmat(temp_path)
        signal = extract_signal(data)

        if signal is None:
            os.unlink(temp_path)
            raise HTTPException(status_code=400, detail="Could not extract signal from .mat file")

        # Segment the signal
        segments = segment_signal(signal)

        if len(segments) == 0:
            os.unlink(temp_path)
            raise HTTPException(status_code=400, detail="Signal too short for segmentation")

        # Prepare data for model
        X = np.array(segments)[..., np.newaxis]  # Add channel dimension

        # Make predictions
        predictions = model.predict(X)
        avg_prediction = np.mean(predictions, axis=0)

        # Get predicted class
        predicted_class_idx = np.argmax(avg_prediction)
        predicted_class = encoder.inverse_transform([predicted_class_idx])[0]

        # Get confidence scores
        confidence_scores = {}
        for i, class_name in enumerate(encoder.classes_):
            confidence_scores[class_name] = float(avg_prediction[i])

        # Clean up
        os.unlink(temp_path)

        return {
            "prediction": predicted_class,
            "confidence_scores": confidence_scores,
            "segments_analyzed": len(segments),
            "signal_length": len(signal)
        }

    except Exception as e:
        # Clean up on error
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        return {"status": "Model not loaded"}
    else:
        return {
            "status": "Model loaded",
            "classes": encoder.classes_.tolist() if encoder else None,
            "input_shape": model.input_shape,
            "output_shape": model.output_shape
        }