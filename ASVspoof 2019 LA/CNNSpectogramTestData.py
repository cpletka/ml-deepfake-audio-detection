#############################################################################################
#  Simple CNN Spoof Detection Model
#
#  This model classifies audio clips as bonafide (real speech) or spoof (synthetic speech)
#  based on their Mel-spectrogram representations.
#
#  Spectrograms were precomputed during preprocessing and stored as .npy files.
#
#  Pipeline:
#     Audio -> Mel Spectrogram -> CNN -> Binary Classification
#
#  The model is trained using the ASVspoof 2019 LA training set and evaluated on the
#  development (dev) set. The best performing model is saved based on dev loss.
#############################################################################################

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#############################################################################################
# Dataset Class
#
# This class loads precomputed Mel-spectrogram features (.npy files) and their labels
# from the ASVspoof protocol file.
#
# Each sample returned consists of:
#    spectrogram tensor of shape (1, 128, 128)
#    binary label (0 = bonafide, 1 = spoof)
#############################################################################################
class PrecomputedASVSpoofDataset(Dataset):

    def __init__(self, protocol_path, feature_dir):

        # Load the protocol file which contains filenames and labels
        self.protocol = pd.read_csv(
            protocol_path,
            sep=r"\s+",
            header=None,
            engine="python"
        )

        # Assign column names
        self.protocol.columns = ['speaker', 'file', 'unused', 'attack', 'label']

        # Convert textual labels to binary values
        self.protocol["label"] = self.protocol["label"].map({
            "bonafide": 0,
            "spoof": 1
        })

        # Directory containing spectrogram .npy files
        self.feature_dir = feature_dir


    # Returns the total number of samples in the dataset
    def __len__(self):
        return len(self.protocol)


    # Loads a single spectrogram and label
    def __getitem__(self, idx):

        row = self.protocol.iloc[idx]
        file_id = row["file"]
        label = row["label"]

        # Construct full path to spectrogram file
        feature_path = os.path.join(self.feature_dir, file_id + ".npy")

        # Load the Mel spectrogram
        mel_db = np.load(feature_path)

        # CNNs expect a channel dimension (like RGB images)
        # Spectrogram becomes shape (1,128,128)
        mel_db = np.expand_dims(mel_db, axis=0)

        return torch.tensor(mel_db, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


#############################################################################################
# CNN Architecture
#
# A simple convolutional neural network used to learn spatial patterns in the spectrogram.
#
# Input shape:
#     (1,128,128)
#
# Layers:
#     Conv -> ReLU -> MaxPool
#     Conv -> ReLU -> MaxPool
#     Flatten -> Dense -> Dropout -> Output
#############################################################################################
class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # Feature extraction layers
        self.features = nn.Sequential(

            # First convolution block
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # Output shape: (16,64,64)

            # Second convolution block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)    # Output shape: (32,32,32)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),   # Helps reduce overfitting
            nn.Linear(128, 1)  # Single output neuron for binary classification
        )

    # Forward pass through the network
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        # Remove extra dimension for compatibility with BCEWithLogitsLoss
        return x.squeeze(1)


#############################################################################################
# Compute Loss on Dev Dataset
#
# This function evaluates model performance on the development set during training.
# It is used to track generalization and determine when to save the best model.
#############################################################################################
def compute_loss(model, data_loader, criterion, device):

    model.eval()
    total_loss = 0.0

    with torch.no_grad():

        for X_batch, y_batch in data_loader:

            # Move data to GPU if available
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            total_loss += loss.item()

    return total_loss / len(data_loader)


#############################################################################################
# Model Evaluation
#
# Computes:
#     Accuracy
#     Precision / Recall / F1-score
#     Confusion Matrix
#############################################################################################
def evaluate_model(model, data_loader, device):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for X_batch, y_batch in data_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)

            # Convert logits to probabilities
            probs = torch.sigmoid(outputs)

            # Apply threshold to obtain binary predictions
            preds = (probs >= 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


#############################################################################################
# Main Training Pipeline
#############################################################################################
def main():

    # Detect whether GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load training dataset
    train_dataset = PrecomputedASVSpoofDataset(
        "./archive/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
        "./processed/train"
    )

    # Load development dataset
    dev_dataset = PrecomputedASVSpoofDataset(
        "./archive/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
        "./processed/dev"
    )

    print("Train samples:", len(train_dataset))
    print("Dev samples:", len(dev_dataset))


    # DataLoader handles batching and shuffling
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )


    # Initialize CNN model
    model = SimpleCNN().to(device)

    # Binary classification loss
    criterion = nn.BCEWithLogitsLoss()

    # Adam optimizer for gradient descent
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20

    # Track best dev loss for checkpoint saving
    best_dev_loss = float("inf")


    # ================================
    # Training Loop
    # ================================
    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            # Backpropagation
            loss.backward()

            optimizer.step()

            running_loss += loss.item()


        avg_train_loss = running_loss / len(train_loader)

        # Evaluate on dev set
        avg_dev_loss = compute_loss(model, dev_loader, criterion, device)


        # Save model checkpoint if dev loss improved
        if avg_dev_loss < best_dev_loss:

            best_dev_loss = avg_dev_loss
            torch.save(model.state_dict(), "best_cnn_model.pth")

            print("Saved best model.")


        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Dev Loss: {avg_dev_loss:.4f}"
        )


    # Load best saved model before final evaluation
    model.load_state_dict(torch.load("best_cnn_model.pth", map_location=device))
    model.to(device)

    print("\nEvaluating on dev set...")

    evaluate_model(model, dev_loader, device)


# Entry point for script execution
if __name__ == "__main__":
    main()