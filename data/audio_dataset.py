import os
import random
import torch
from torch.utils.data import Dataset
import torchaudio
import numpy as np
from utils.audio_utils import pad_audio, parse_labels


class AudioDataset(Dataset):
    def __init__(self, audio_files, label_dir, sample_rate=16000, target_length=6):
        self.audio_files = audio_files
        self.label_dir = label_dir
        self.sample_rate = sample_rate
        self.target_length = target_length * sample_rate
        self.raw_target_length = target_length

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        try:
            waveform, sr = torchaudio.load(audio_path)
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
            waveform = pad_audio(waveform, self.sample_rate, self.raw_target_length)

            audio_filename = os.path.basename(audio_path).replace(".wav", "")
            if audio_filename.startswith("RFP_R"):
                labels = np.zeros(int(self.raw_target_length / 0.010), dtype=np.float32)
            else:
                label_path = os.path.join(self.label_dir, f"{audio_filename}.wav_labels.txt")
                labels = parse_labels(label_path, self.raw_target_length, self.sample_rate).astype(np.float32)

            return waveform, torch.tensor(labels, dtype=torch.float32)
        
        except (OSError, IOError) as e:
            print(f"Error opening file {audio_path}: {e}")
            new_idx = random.randint(0, len(self.audio_files) - 1)
            return self.__getitem__(new_idx)


def get_audio_file_paths(extrinsic_dir, intrinsic_dir, real_dir):
    extrinsic_files = [os.path.join(extrinsic_dir, f) for f in os.listdir(extrinsic_dir)
                       if f.endswith(".wav") and not f.startswith("partial_fake")]
    intrinsic_files = [os.path.join(intrinsic_dir, f) for f in os.listdir(intrinsic_dir)
                       if f.endswith(".wav") and not f.startswith("partial_fake")]
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir)
                  if f.endswith(".wav") and not f.startswith("partial_fake")]
    
    audio_files = [f for f in extrinsic_files + real_files 
                   if os.path.basename(f).startswith(("extrinsic"))]
    return audio_files
