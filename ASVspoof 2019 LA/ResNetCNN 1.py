#############################################################################################
#  Residual CNN (Small ResNet) for Spoof Detection
#
#  This model classifies audio clips as bonafide (real speech) or spoof (synthetic speech)
#  using Mel-spectrogram representations of audio signals.
#
#  Spectrograms are precomputed during preprocessing and stored as .npy files.
#
#  Pipeline:
#       Audio -> Mel Spectrogram -> Residual CNN -> Binary Classification
#
#  This model uses residual blocks inspired by ResNet architecture to improve gradient
#  flow and allow deeper feature extraction compared to a simple CNN.
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
# Loads spectrogram features (.npy files) and labels from the ASVspoof protocol file.
#
# Each returned sample contains:
#     spectrogram tensor (1,128,128)
#     binary label (0 = bonafide, 1 = spoof)
#############################################################################################
class PrecomputedASVSpoofDataset(Dataset):

    def __init__(self, protocol_path, feature_dir):

        # Load the ASVspoof protocol file containing filenames and labels
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

        # Directory where spectrogram files are stored
        self.feature_dir = feature_dir


    # Return number of samples in dataset
    def __len__(self):
        return len(self.protocol)


    # Load a single spectrogram and label
    def __getitem__(self, idx):

        row = self.protocol.iloc[idx]
        file_id = row["file"]
        label = row["label"]

        # Load spectrogram file
        feature_path = os.path.join(self.feature_dir, file_id + ".npy")
        mel_db = np.load(feature_path)

        # Add channel dimension required by CNNs
        # Final shape becomes (1,128,128)
        mel_db = np.expand_dims(mel_db, axis=0)

        return torch.tensor(mel_db, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


#############################################################################################
# Residual Block
#
# Instead of learning a direct mapping, the network learns a residual function:
#       output = F(x) + x
#
# This helps avoid vanishing gradients and allows deeper networks to train effectively.
#############################################################################################
class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # First convolution layer
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Second convolution layer
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        # If dimensions change, a 1x1 convolution adjusts them
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )


    def forward(self, x):

        # Save input for residual connection
        identity = self.shortcut(x)

        # First convolution block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second convolution block
        out = self.conv2(out)
        out = self.bn2(out)

        # Add residual connection
        out += identity

        # Final activation
        out = self.relu(out)

        return out


#############################################################################################
# Small ResNet Architecture
#
# A lightweight ResNet designed for spectrogram classification.
#
# Structure:
#   Stem layer
#   Residual Block Group 1
#   Residual Block Group 2
#   Residual Block Group 3
#   Global Average Pool
#   Fully Connected Classifier
#############################################################################################
class SmallResNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Initial convolution layer (feature extraction start)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Residual layer group 1
        self.layer1 = nn.Sequential(
            ResidualBlock(16, 16),
            ResidualBlock(16, 16)
        )

        # Residual layer group 2 (downsampling)
        self.layer2 = nn.Sequential(
            ResidualBlock(16, 32, stride=2),
            ResidualBlock(32, 32)
        )

        # Residual layer group 3 (downsampling)
        self.layer3 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64)
        )

        # Global pooling reduces feature maps to 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout helps prevent overfitting
        self.dropout = nn.Dropout(0.3)

        # Final binary classifier
        self.fc = nn.Linear(64, 1)


    def forward(self, x):

        # Initial convolution
        x = self.stem(x)       # (B,16,128,128)

        # Residual blocks
        x = self.layer1(x)     # (B,16,128,128)
        x = self.layer2(x)     # (B,32,64,64)
        x = self.layer3(x)     # (B,64,32,32)

        # Global average pooling
        x = self.pool(x)       # (B,64,1,1)

        # Flatten features
        x = torch.flatten(x, 1)

        x = self.dropout(x)

        # Final classification layer
        x = self.fc(x)

        return x.squeeze(1)


#############################################################################################
# Compute validation loss during training
#############################################################################################
def compute_loss(model, data_loader, criterion, device):

    model.eval()
    total_loss = 0.0

    with torch.no_grad():

        for X_batch, y_batch in data_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            total_loss += loss.item()

    return total_loss / len(data_loader)


#############################################################################################
# Evaluate final model performance
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

            # Convert probabilities to binary predictions
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

    # Detect GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load datasets
    train_dataset = PrecomputedASVSpoofDataset(
        "./archive/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
        "./processed/train"
    )

    dev_dataset = PrecomputedASVSpoofDataset(
        "./archive/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
        "./processed/dev"
    )

    print("Train samples:", len(train_dataset))
    print("Dev samples:", len(dev_dataset))

    # Data loaders for batching
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

    # Initialize model
    model = SmallResNet().to(device)

    # Binary classification loss
    criterion = nn.BCEWithLogitsLoss()

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20

    # Track best dev loss for saving model
    best_dev_loss = float("inf")


    # ============================
    # Training Loop
    # ============================
    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Compute validation loss
        avg_dev_loss = compute_loss(model, dev_loader, criterion, device)

        # Save best model checkpoint
        if avg_dev_loss < best_dev_loss:
            best_dev_loss = avg_dev_loss
            torch.save(model.state_dict(), "best_resnet_model.pth")
            print("Saved best model.")

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Dev Loss: {avg_dev_loss:.4f}"
        )

    # Reload best model before evaluation
    model.load_state_dict(torch.load("best_resnet_model.pth", map_location=device))
    model.to(device)

    print("\nEvaluating on dev set...")

    evaluate_model(model, dev_loader, device)


if __name__ == "__main__":
    main()