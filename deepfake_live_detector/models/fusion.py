import torch
import torch.nn as nn

class EvidenceFusion(nn.Module):
    """
    Stage 7: Evidence Fusion (Gated Mixture of Experts).
    Combines features from the RGB, Noise, Frequency, and Semantic branches.
    Uses quality indicators (like metadata availability, blur score) to gate features.
    """
    
    def __init__(self, spatial_dim, noise_dim, freq_dim, semantic_dim, quality_dim, num_classes=5):
        super().__init__()
        
        # We concatenate the features: spatial + noise + freq + semantic
        self.combined_dim = spatial_dim + noise_dim + freq_dim + semantic_dim
        
        # Gating network based on quality indicators
        self.gating_network = nn.Sequential(
            nn.Linear(quality_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.combined_dim),
            nn.Sigmoid()  # Outputs attention weights for the combined feature vector
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, features: dict, quality: torch.Tensor):
        """
        features: dict containing 'spatial', 'noise', 'freq', 'semantic' tensors
        quality: tensor of quality indicators (e.g., [is_blurry, has_metadata, has_c2pa, jpeg_quality])
        """
        # Concatenate features
        combined_features = torch.cat([
            features["spatial"], 
            features["noise"], 
            features["freq"], 
            features["semantic"]
        ], dim=1)
        
        # Compute gates
        gates = self.gating_network(quality)
        
        # Apply gates
        gated_features = combined_features * gates
        
        # Classify
        logits = self.classifier(gated_features)
        
        return logits, gated_features
