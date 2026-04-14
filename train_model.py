import scipy.io
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

print("Starting training script...")

def extract_signal(data):
    for key in data:
        if "DE_time" in key:
            return data[key].flatten()
    return None

def segment_signal(signal, window_size=2048, step=1024):
    segments = []
    for start in range(0, len(signal) - window_size, step):
        seg = signal[start:start + window_size]
        # Normalize each segment
        seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-8)
        segments.append(seg)
    return segments

def create_model():
    model = Sequential([
        Conv1D(16, kernel_size=3, activation='relu', input_shape=(2048,1)),
        MaxPooling1D(2),
        Conv1D(32, kernel_size=3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(4, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    print("Loading data...")
    base_path = "Machine_project"

    X = []
    y = []

    # Normal data
    normal_path = os.path.join(base_path, "Normal")
    print(f"Loading normal data from {normal_path}")
    for file in os.listdir(normal_path):
        if file.endswith(".mat"):
            file_path = os.path.join(normal_path, file)
            try:
                data = scipy.io.loadmat(file_path)
                signal = extract_signal(data)
                if signal is not None:
                    segments = segment_signal(signal)
                    for seg in segments:
                        X.append(seg)
                        y.append("Normal")
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue

    # Fault data
    fault_path = os.path.join(base_path, "s-whynot CWRU-dataset main 12k_Drive_End_Bearing_Fault_Data")
    print(f"Loading fault data from {fault_path}")
    for fault_type in os.listdir(fault_path):
        fault_type_path = os.path.join(fault_path, fault_type)
        if os.path.isdir(fault_type_path):
            for root, dirs, files in os.walk(fault_type_path):
                for file in files:
                    if file.endswith(".mat"):
                        file_path = os.path.join(root, file)
                        try:
                            data = scipy.io.loadmat(file_path)
                            signal = extract_signal(data)
                            if signal is not None:
                                segments = segment_signal(signal)
                                for seg in segments:
                                    X.append(seg)
                                    y.append(fault_type)
                        except Exception as e:
                            print(f"Error loading {file}: {e}")
                            continue

    print(f"Total samples: {len(X)}")

    if len(X) == 0:
        print("No data loaded!")
        return

    # Encode labels
    X = np.array(X)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Save encoder classes
    np.save('label_classes.npy', encoder.classes_)
    print(f"Classes: {encoder.classes_}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Add channel dimension
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    print(f"Train shape: {X_train.shape}")

    # Train model
    print("Training model...")
    model = create_model()
    model.fit(
        X_train, y_train,
        epochs=5,  # Reduced for testing
        batch_size=32,
        validation_data=(X_test, y_test)
    )

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Model accuracy: {accuracy}")

    # Save model
    model.save('bearing_fault_model.h5')
    print("Model saved as bearing_fault_model.h5")

if __name__ == "__main__":
    main()