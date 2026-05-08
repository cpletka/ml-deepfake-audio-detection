# Logistic Regression Model for ASVspoof 2019 LA Dataset (development set)
# Final version of the logistic regression model for the ASVspoof 2019 LA dataset.
# This code extracts MFCCs and additional spectral features, then trains and evaluates a logistic regression classifier on the development set.

import pandas as pd
import numpy as np
import librosa
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib


def load_protocol(protocol_path):
    protocol = pd.read_csv(
        protocol_path,
        sep=" ",
        header=None,
        engine="python"
    )
    protocol.columns = ['speaker', 'file', 'unused', 'attack', 'label']
    protocol["label"] = protocol["label"].map({
        "bonafide": 0,
        "spoof": 1
    })
    return protocol

## Extract features from the audio files
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000)

    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)

    # Additional spectral features
    zcr = librosa.feature.zero_crossing_rate(audio)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    rms = librosa.feature.rms(y=audio)

    # Mean and std for each added feature
    extra_features = np.array([
        np.mean(zcr), np.std(zcr),
        np.mean(spectral_centroid), np.std(spectral_centroid),
        np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
        np.mean(spectral_rolloff), np.std(spectral_rolloff),
        np.mean(rms), np.std(rms)
    ])

    # Combine everything
    features = np.concatenate((mfcc_mean, mfcc_std, extra_features))

    return features

def build_dataset(protocol, audio_dir):
    X = []
    y = []

    for _, row in protocol.iterrows():
        file_id = row['file']
        label = row['label']

        if not file_id.endswith(".flac"):
            file_id += ".flac"

        file_path = os.path.join(audio_dir, file_id)

        try:
            features = extract_features(file_path)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return np.array(X), np.array(y)

# Load train protocol and dev protocol
train_protocol = load_protocol(
    "./archive/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
)

dev_protocol = load_protocol(
    "./archive/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
)

# Build train and dev datasets
X_train, y_train = build_dataset(
    train_protocol,
    "./archive/LA/LA/ASVspoof2019_LA_train/flac"
)

X_dev, y_dev = build_dataset(
    dev_protocol,
    "./archive/LA/LA/ASVspoof2019_LA_dev/flac"
)

print("Train shape:", X_train.shape, y_train.shape)
print("Dev shape:", X_dev.shape, y_dev.shape)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_dev = scaler.transform(X_dev)

# Train model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Predict on dev set
y_pred = model.predict(X_dev)

# Evaluate
accuracy = accuracy_score(y_dev, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_dev, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_dev, y_pred))

# Save the model and scaler for future use
joblib.dump(model, "logistic_regression_model.pkl")
joblib.dump(scaler, "feature_scaler.pkl")