import os
import numpy as np
import pandas as pd
import librosa

def load_protocol(protocol_path):
    protocol = pd.read_csv(
        protocol_path,
        sep=r"\s+",
        header=None,
        engine="python"
    )
    protocol.columns = ['speaker', 'file', 'unused', 'attack', 'label']
    return protocol

def create_melspec(file_path, target_frames=128):
    audio, sr = librosa.load(file_path, sr=16000)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=128
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < target_frames:
        pad_width = target_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :target_frames]

    return mel_db.astype(np.float32)

def preprocess_dataset(protocol_path, audio_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    protocol = load_protocol(protocol_path)

    for _, row in protocol.iterrows():
        file_id = row["file"]

        if not file_id.endswith(".flac"):
            file_id += ".flac"

        input_path = os.path.join(audio_dir, file_id)
        output_path = os.path.join(output_dir, file_id.replace(".flac", ".npy"))

        try:
            mel_db = create_melspec(input_path)
            np.save(output_path, mel_db)
        except Exception as e:
            print(f"Error processing {input_path}: {e}")

# Train set
preprocess_dataset(
    "./archive/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
    "./archive/LA/LA/ASVspoof2019_LA_train/flac",
    "./processed/train"
)

# Dev set
preprocess_dataset(
    "./archive/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
    "./archive/LA/LA/ASVspoof2019_LA_dev/flac",
    "./processed/dev"
)