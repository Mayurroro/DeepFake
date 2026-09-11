import torch
import torch.nn as nn


class AudioDetector(nn.Module):
    """
    Enhanced CNN-LSTM-Attention hybrid for deepfake audio detection on Mel-spectrograms.
    Auxiliary dense branch ingests extended spectral statistics.
    """

    def __init__(self, num_classes=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        # After 3 × pool2: n_mels 128→16, features per time-step = 64*16 = 1024
        self.lstm = nn.LSTM(input_size=1024, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        
        # Self-attention over LSTM outputs (bidirectional = 256)
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True, dropout=0.3)
        
        # Expanded auxiliary branch
        self.aux_dense = nn.Sequential(
            nn.Linear(30, 64), 
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), 
            nn.Linear(256 + 32, 128), 
            nn.ReLU(), 
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, mel, aux=None):
        c = self.cnn(mel)                        # (B,64,16,T//8)
        c = c.permute(0, 3, 1, 2).flatten(2)    # (B,T//8,1024)
        
        out, _ = self.lstm(c)                    # out: (B, T//8, 256)
        
        # Apply self-attention
        attn_out, _ = self.attention(out, out, out) # (B, T//8, 256)
        
        # Global average pooling over time
        h = attn_out.mean(dim=1)                 # (B, 256)
        
        parts = [h]
        if aux is not None:
            parts.append(self.aux_dense(aux))
        else:
            parts.append(torch.zeros(mel.size(0), 32, device=mel.device))
            
        logits = self.classifier(torch.cat(parts, 1))
        return logits, torch.cat(parts, 1)
