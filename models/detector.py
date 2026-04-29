"""
Deepfake Detection Model
========================
Uses EfficientNet + Attention mechanism for binary classification.
Supports image and video frame analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from typing import Tuple, Dict, Optional


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention module."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial attention module for artifact localization."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, channels: int):
        super().__init__()
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class FrequencyBranch(nn.Module):
    """
    Frequency-domain analysis branch.
    Detects GAN artifacts in DCT/FFT domain.
    """

    def __init__(self, out_features: int = 256):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Linear(128 * 16, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to grayscale and compute FFT magnitude
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        fft = torch.fft.fft2(gray)
        magnitude = torch.log(torch.abs(fft) + 1e-8).unsqueeze(1)
        feat = self.conv_layers(magnitude)
        feat = feat.view(feat.size(0), -1)
        return F.relu(self.fc(feat))


class DeepfakeDetector(nn.Module):
    """
    Multi-branch deepfake detection network combining:
    1. EfficientNet-B4 backbone (spatial features)
    2. Frequency analysis branch (GAN fingerprints)
    3. CBAM attention for artifact localization
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.4):
        super().__init__()

        # --- Spatial Branch (EfficientNet-B4 backbone) ---
        efficientnet = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        )
        # Remove classifier
        self.backbone = nn.Sequential(*list(efficientnet.children())[:-2])
        spatial_features = 1792  # EfficientNet-B4 output channels

        # Attention after backbone
        self.cbam = CBAM(spatial_features)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        # --- Frequency Branch ---
        freq_features = 256
        self.freq_branch = FrequencyBranch(out_features=freq_features)

        # --- Fusion + Classifier ---
        fused = spatial_features + freq_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Spatial branch
        spatial = self.backbone(x)
        spatial = self.cbam(spatial)
        spatial_feat = self.spatial_pool(spatial).view(x.size(0), -1)

        # Frequency branch
        freq_feat = self.freq_branch(x)

        # Fuse and classify
        fused = torch.cat([spatial_feat, freq_feat], dim=1)
        logit = self.classifier(fused)
        prob = torch.sigmoid(logit).squeeze(1)

        return {
            "logit": logit.squeeze(1),
            "probability": prob,
            "prediction": (prob > 0.5).long(),
        }

    def get_gradcam_target_layer(self):
        """Return target layer for Grad-CAM visualization."""
        return self.backbone[-1]


class EnsembleDetector:
    """
    Ensemble of multiple detector models for improved accuracy.
    Uses soft voting (averaged probabilities).
    """

    def __init__(self, model_paths: list, device: str = "cpu"):
        self.device = device
        self.models = []
        for path in model_paths:
            m = DeepfakeDetector(pretrained=False)
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            self.models.append(m)

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        probs = []
        with torch.no_grad():
            for model in self.models:
                out = model(x)
                probs.append(out["probability"])
        avg_prob = torch.stack(probs).mean(0)
        return {
            "probability": avg_prob,
            "prediction": (avg_prob > 0.5).long(),
            "confidence": torch.abs(avg_prob - 0.5) * 2,
        }


def build_model(
    pretrained: bool = True, device: str = "cpu"
) -> Tuple[DeepfakeDetector, torch.device]:
    """Factory function to build and return the model."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = DeepfakeDetector(pretrained=pretrained)
    model = model.to(device)
    return model, device
