# Bearing Fault Detection System

A complete machine learning system for detecting bearing faults using vibration data from .mat files. Includes a web UI (React) and mobile app (Flutter) that interface with a FastAPI backend.

## Project Structure

```
Machine_fault_detection/
├── Machine_project/           # Dataset folder
├── backend/                   # FastAPI backend
│   ├── main.py               # API server
│   └── requirements.txt      # Python dependencies
├── frontend/                  # React web application
│   ├── public/
│   ├── src/
│   └── package.json
├── mobile/                    # Flutter mobile app
│   ├── lib/
│   └── pubspec.yaml
├── bearing_fault_model.h5     # Trained ML model
├── label_classes.npy          # Label encoder classes
├── train_model.py            # Model training script
└── README.md
```

## Features

- **CNN-based Classification**: Trained on CWRU bearing dataset
- **4 Classes**: Normal, Ball fault (B), Inner Race fault (IR), Outer Race fault (OR)
- **Web Interface**: React-based file upload and results display
- **Mobile App**: Flutter app for Android/iOS
- **Real-time Prediction**: Fast inference on uploaded .mat files

## Model Performance

- **Accuracy**: 95.81% on test set
- **Architecture**: 1D CNN with Conv1D layers
- **Input**: 2048-sample vibration signal segments
- **Classes**: ['B', 'IR', 'Normal', 'OR']

## Setup Instructions

### 1. Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 2. Web Frontend Setup

```bash
cd frontend
npm install
npm start
```

The web app will be available at `http://localhost:3000`

### 3. Mobile App Setup

#### Prerequisites:

- Flutter SDK installed
- Android Studio / Xcode for device simulation

```bash
cd mobile
flutter pub get
flutter run
```

For Android emulator, the API URL is configured as `http://10.0.2.2:8000`
For iOS simulator, use `http://localhost:8000`

## API Endpoints

### POST /predict

Upload a .mat file for prediction

**Request**: Multipart form data with 'file' field
**Response**:

```json
{
  "prediction": "Normal",
  "confidence_scores": {
    "Normal": 0.95,
    "B": 0.02,
    "IR": 0.02,
    "OR": 0.01
  },
  "segments_analyzed": 45,
  "signal_length": 120000
}
```

### GET /model-info

Get information about the loaded model

## Usage

1. Start the backend server
2. Start the web frontend or mobile app
3. Upload a .mat file containing bearing vibration data
4. View the prediction results with confidence scores

## Dataset

The system is trained on the CWRU Bearing Dataset:

- Normal bearings
- Faulty bearings with different fault types and severities
- Vibration signals sampled at 12kHz drive end

## Technologies Used

- **Backend**: FastAPI, TensorFlow, SciPy
- **Web Frontend**: React, Axios
- **Mobile**: Flutter, HTTP
- **ML**: TensorFlow/Keras CNN

## Notes

- The model expects .mat files with 'DE_time' key containing vibration signals
- Signals are segmented into 2048-sample windows with 50% overlap
- Each segment is normalized before prediction
- Results show average prediction across all segments
