import torch
import numpy as np


def pad_audio(audio, sample_rate=16000, target_duration=6):
    target_length = int(sample_rate * target_duration)
    current_length = audio.shape[1]
    
    if current_length < target_length:
        padding = target_length - current_length
        audio = torch.cat((audio, torch.zeros(audio.shape[0], padding)), dim=1)
    elif current_length > target_length:
        if current_length - target_length == 1:
            audio = torch.cat((audio, torch.zeros(audio.shape[0], 1)), dim=1)
        else:
            audio = audio[:, :target_length]
    
    return audio


def parse_labels(file_path, audio_length, sample_rate, frame_duration=0.02):
    frames_per_audio = int(audio_length / frame_duration)
    labels = np.zeros(frames_per_audio, dtype=np.float32)

    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            start, end, authenticity = line.strip().split('-')
            start_time = float(start)
            end_time = float(end)

            if authenticity == 'F':
                start_frame = int(start_time / frame_duration)
                end_frame = int(end_time / frame_duration)
                labels[start_frame:end_frame] = 1
                
                for offset in range(1, 5):
                    if start_frame - offset >= 0:
                        labels[start_frame - offset] = 1
                    if end_frame + offset < frames_per_audio:
                        labels[end_frame + offset] = 1

    return labels
