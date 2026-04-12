import torch
import torch.nn as nn


class AudioDetector(nn.Module):
    """CNN-LSTM hybrid for deepfake audio detection on Mel-spectrograms.
    Auxiliary dense branch ingests spectral statistics (centroid, rolloff, MFCCs).
    """

    def __init__(self, num_classes=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        # After 3 × pool2: n_mels 128→16, features per time-step = 64*16 = 1024
        self.lstm = nn.LSTM(input_size=1024, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3)
        self.aux_dense = nn.Sequential(nn.Linear(30, 32), nn.ReLU())  # 2+2+13+13 = 30
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(160, 64), nn.ReLU(), nn.Linear(64, num_classes),
        )

    def forward(self, mel, aux=None):
        c = self.cnn(mel)                        # (B,64,16,T//8)
        c = c.permute(0, 3, 1, 2).flatten(2)    # (B,T//8,1024)
        out, _ = self.lstm(c)
        h = out[:, -1, :]                        # last hidden
        parts = [h]
        parts.append(self.aux_dense(aux) if aux is not None else torch.zeros(mel.size(0), 32, device=mel.device))
        logits = self.classifier(torch.cat(parts, 1))
        return logits, torch.cat(parts, 1)
