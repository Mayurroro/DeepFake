import torch
import torch.nn as nn
import timm


class ImageDetector(nn.Module):
    """EfficientNet-B4 backbone with multi-head attention for deepfake image detection.
    Fuses RGB features with FFT frequency maps and Canny edge features.
    Output: 3-class (REAL=0, MANIPULATED=1, AI_GENERATED=2) + feature vector for ensemble.
    """

    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b4", pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features  # 1792

        # Lightweight heads for auxiliary signals
        self.freq_head = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.edge_head = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )

        combined = feat_dim + 64  # 1792 + 32 + 32
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(combined, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, num_classes),
        )

    def forward(self, rgb, freq=None, edges=None):
        feat = self.backbone(rgb)
        parts = [feat]
        parts.append(self.freq_head(freq) if freq is not None else torch.zeros(rgb.size(0), 32, device=rgb.device))
        parts.append(self.edge_head(edges) if edges is not None else torch.zeros(rgb.size(0), 32, device=rgb.device))
        combined = torch.cat(parts, dim=1)
        logits = self.classifier(combined)
        return logits, combined
