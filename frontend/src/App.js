import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.name.endsWith('.mat')) {
      setSelectedFile(file);
      setError(null);
      setPrediction(null);
    } else {
      setError('Please select a .mat file');
      setSelectedFile(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('http://localhost:8000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setPrediction(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const renderConfidenceBar = (className, confidence) => {
    const percentage = (confidence * 100).toFixed(1);
    return (
      <div className="confidence-item">
        <div className="confidence-label">
          <span>{className}</span>
          <span>{percentage}%</span>
        </div>
        <div className="confidence-bar">
          <div
            className="confidence-fill"
            style={{ width: `${percentage}%` }}
          ></div>
        </div>
      </div>
    );
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Bearing Fault Detection</h1>
        <p>Upload a .mat file to analyze bearing vibration data</p>
      </header>

      <main className="App-main">
        <div className="upload-section">
          <div className="file-input-container">
            <input
              type="file"
              accept=".mat"
              onChange={handleFileSelect}
              id="file-input"
              className="file-input"
            />
            <label htmlFor="file-input" className="file-input-label">
              Choose .mat file
            </label>
            {selectedFile && (
              <div className="selected-file">
                Selected: {selectedFile.name}
              </div>
            )}
          </div>

          <button
            onClick={handleUpload}
            disabled={!selectedFile || loading}
            className="upload-button"
          >
            {loading ? 'Analyzing...' : 'Analyze Bearing'}
          </button>
        </div>

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {prediction && (
          <div className="results-section">
            <h2>Analysis Results</h2>

            <div className="prediction-result">
              <div className="main-prediction">
                <h3>Predicted Condition:</h3>
                <div className={`prediction-badge ${prediction.prediction.toLowerCase()}`}>
                  {prediction.prediction}
                </div>
              </div>

              <div className="signal-info">
                <p>Signal Length: {prediction.signal_length.toLocaleString()} samples</p>
                <p>Segments Analyzed: {prediction.segments_analyzed}</p>
              </div>
            </div>

            <div className="confidence-section">
              <h3>Confidence Scores</h3>
              <div className="confidence-bars">
                {Object.entries(prediction.confidence_scores).map(([className, confidence]) =>
                  renderConfidenceBar(className, confidence)
                )}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;