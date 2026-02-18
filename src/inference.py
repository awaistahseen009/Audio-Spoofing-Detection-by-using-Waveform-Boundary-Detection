import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torchaudio
import numpy as np
import soundfile as sf
from src.model import BoundaryDetectionModel
from utils.audio_utils import pad_audio
from utils.logger import setup_logger

logger = setup_logger(__name__, log_type='inference')


def load_model(checkpoint_path, device):
    model = BoundaryDetectionModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Model loaded from {checkpoint_path}")
    return model


def preprocess_audio(audio_path, sample_rate=16000, target_length=6):
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim == 1:
        data = data[np.newaxis, :]
    else:
        data = data.T
    waveform = torch.tensor(data)
    waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    waveform = pad_audio(waveform, sample_rate, target_length)
    return waveform


def infer_single_audio(model, audio_path, device, threshold=0.5):
    logger.info(f"Processing audio: {audio_path}")
    audio_tensor = preprocess_audio(audio_path).to(device)

    with torch.no_grad():
        output = model(audio_tensor).squeeze(-1).cpu().numpy()
        prediction = (output > threshold).astype(int)

    prediction_flat = prediction.flatten()  # collapse batch dim to get true frame count
    output_flat = output.flatten()

    fake_percentage = float((prediction_flat.sum() / len(prediction_flat)) * 100)
    logger.info(f"Fake frames detected: {fake_percentage:.2f}%")

    return {
        "output": output_flat.tolist(),
        "prediction": prediction_flat.tolist(),
        "fake_percentage": fake_percentage,
        "total_frames": int(len(prediction_flat)),
        "fake_frames": int(prediction_flat.sum())
    }


def main_inference(audio_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = load_model(checkpoint_path, device)
    result = infer_single_audio(model, audio_path, device)

    logger.info(f"Inference completed - Fake: {result['fake_percentage']:.2f}%")
    return result


if __name__ == "__main__":
    audio_path = "Real/RFP_R_24918.wav"
    checkpoint_path = "checkpoints/checkpoint_epoch_4.pt"
    main_inference(audio_path, checkpoint_path)