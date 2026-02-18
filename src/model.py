import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        config = Wav2Vec2Config.from_pretrained(
            "facebook/wav2vec2-base-960h",
            local_files_only=True
        )
        self.wav2vec2 = Wav2Vec2Model(config)

    def forward(self, audio):
        audio = audio.squeeze(1)
        features = self.wav2vec2(audio).last_hidden_state
        return features


class FrameLevelEmbedding(nn.Module):
    def __init__(self):
        super(FrameLevelEmbedding, self).__init__()
        self.cnn1 = nn.Conv1d(in_channels=768, out_channels=512, kernel_size=5, padding=2)
        self.res_blocks = nn.Sequential(*[ResBlock(512) for _ in range(12)])
        self.cnn2 = nn.Conv1d(in_channels=512, out_channels=128, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn1(x)
        x = self.res_blocks(x)
        x = self.cnn2(x)
        x = x.transpose(1, 2)
        return x


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class FrameLevelClassifier(nn.Module):
    def __init__(self):
        super(FrameLevelClassifier, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=1024, batch_first=True), num_layers=2
        )
        self.bilstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = self.transformer(x)
        x, _ = self.bilstm(x)
        x = self.fc(x)
        return torch.sigmoid(x)


class BoundaryDetectionModel(nn.Module):
    def __init__(self):
        super(BoundaryDetectionModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.frame_embedding = FrameLevelEmbedding()
        self.classifier = FrameLevelClassifier()

    def forward(self, audio):
        features = self.feature_extractor(audio)
        embeddings = self.frame_embedding(features)
        output = self.classifier(embeddings)
        return output
