import torch
import torch.nn as nn
import timm

from .fusion import EvidenceFusion


class ImageDetector(nn.Module):
    """
    Stage 6: Learned Detectors Ensemble.
    Contains branches for RGB, Noise, Frequency, and Semantic features.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        
        # 1. Spatial/RGB Branch (EfficientNet)
        self.spatial_branch = timm.create_model("efficientnet_b4", pretrained=pretrained, num_classes=0)
        self.spatial_dim = self.spatial_branch.num_features
        
        # 2. Noise/Residual Branch (Placeholder for constrained convolution network)
        self.noise_branch = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        self.noise_dim = 32
        
        # 3. Frequency Branch (FFT/DCT)
        self.freq_branch = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.freq_dim = 32
        
        # 4. Semantic/Geometry Branch (Placeholder)
        self.semantic_branch = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 64) # Dummy input size for semantic features
        )
        self.semantic_dim = 64

    def forward(self, rgb, noise=None, freq=None, semantic=None):
        features = {}
        
        features["spatial"] = self.spatial_branch(rgb)
        
        if noise is not None:
            features["noise"] = self.noise_branch(noise)
        else:
            features["noise"] = torch.zeros(rgb.size(0), self.noise_dim, device=rgb.device)
            
        if freq is not None:
            features["freq"] = self.freq_branch(freq)
        else:
            features["freq"] = torch.zeros(rgb.size(0), self.freq_dim, device=rgb.device)
            
        if semantic is not None:
            features["semantic"] = self.semantic_branch(semantic)
        else:
            features["semantic"] = torch.zeros(rgb.size(0), self.semantic_dim, device=rgb.device)
            
        return features


class ImageDetectorEnsemble(nn.Module):
    """End-to-end learned detector (Stages 6-7): multi-branch features + gated fusion classifier.

    Combines the ImageDetector feature extractor with the EvidenceFusion head so the
    whole ensemble trains in one pass. quality is a per-image quality vector (blur,
    metadata presence, etc.); zeros fall back to a learned constant gate.
    """

    def __init__(self, pretrained=True, num_classes=3, quality_dim=4):
        super().__init__()
        self.branches = ImageDetector(pretrained=pretrained)
        self.fusion = EvidenceFusion(
            self.branches.spatial_dim, self.branches.noise_dim,
            self.branches.freq_dim, self.branches.semantic_dim,
            quality_dim=quality_dim, num_classes=num_classes,
        )

    def forward(self, rgb, noise=None, freq=None, semantic=None, quality=None):
        feats = self.branches(rgb, noise, freq, semantic)
        if quality is None:
            quality = torch.zeros(rgb.size(0), self.fusion.gating_network[0].in_features,
                                  device=rgb.device)
        logits, gated_features = self.fusion(feats, quality)
        return logits, gated_features
