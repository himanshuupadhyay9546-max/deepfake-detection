"""
Unit Tests for Deepfake Detection System
"""

import io
import pytest
import torch
import numpy as np
from PIL import Image

# ─────────────────────────────────────────
#  Model Tests
# ─────────────────────────────────────────

def make_dummy_batch(batch_size=2, size=224):
    return torch.randn(batch_size, 3, size, size)


class TestDeepfakeDetector:
    def test_forward_pass(self):
        from models.detector import DeepfakeDetector
        model = DeepfakeDetector(pretrained=False)
        model.eval()
        x = make_dummy_batch(2)
        with torch.no_grad():
            out = model(x)
        assert "probability" in out
        assert "prediction" in out
        assert out["probability"].shape == (2,)
        assert ((out["probability"] >= 0) & (out["probability"] <= 1)).all()

    def test_predictions_binary(self):
        from models.detector import DeepfakeDetector
        model = DeepfakeDetector(pretrained=False)
        model.eval()
        x = make_dummy_batch(4)
        with torch.no_grad():
            out = model(x)
        assert set(out["prediction"].unique().tolist()).issubset({0, 1})

    def test_channel_attention(self):
        from models.detector import ChannelAttention
        attn = ChannelAttention(64)
        x = torch.randn(2, 64, 7, 7)
        out = attn(x)
        assert out.shape == x.shape

    def test_spatial_attention(self):
        from models.detector import SpatialAttention
        attn = SpatialAttention()
        x = torch.randn(2, 64, 7, 7)
        out = attn(x)
        assert out.shape == x.shape

    def test_frequency_branch(self):
        from models.detector import FrequencyBranch
        fb = FrequencyBranch(out_features=128)
        x = torch.randn(2, 3, 224, 224)
        out = fb(x)
        assert out.shape == (2, 128)

    def test_build_model(self):
        from models.detector import build_model
        model, device = build_model(pretrained=False, device="cpu")
        assert model is not None
        assert device is not None


# ─────────────────────────────────────────
#  Inference Tests
# ─────────────────────────────────────────

class TestInference:
    def _make_dummy_image(self) -> Image.Image:
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        return Image.fromarray(arr)

    def test_predict_image(self):
        from src.inference import DeepfakeInference
        engine = DeepfakeInference(generate_heatmap=False)
        img = self._make_dummy_image()
        result = engine.predict_image(img, crop_face=False)
        assert result.label in {"REAL", "FAKE"}
        assert 0 <= result.probability <= 1
        assert 0 <= result.confidence <= 1

    def test_result_to_dict(self):
        from src.inference import DetectionResult
        r = DetectionResult(label="FAKE", probability=0.8, confidence=0.6)
        d = r.to_dict()
        assert d["label"] == "FAKE"
        assert d["is_fake"] is True

    def test_batch_predict(self):
        from src.inference import DeepfakeInference
        engine = DeepfakeInference(generate_heatmap=False)
        images = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        results = engine.predict_batch(images, crop_face=False)
        assert len(results) == 3
        for r in results:
            assert r.label in {"REAL", "FAKE"}


# ─────────────────────────────────────────
#  Metrics Tests
# ─────────────────────────────────────────

class TestMetrics:
    def test_compute_metrics(self):
        from utils.evaluate import compute_metrics
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        m = compute_metrics(y_true, y_prob)
        assert "auc" in m
        assert "f1" in m
        assert 0 <= m["auc"] <= 1

    def test_optimal_threshold(self):
        from utils.evaluate import find_optimal_threshold
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        t = find_optimal_threshold(y_true, y_prob)
        assert 0 <= t <= 1


# ─────────────────────────────────────────
#  Preprocessing Tests
# ─────────────────────────────────────────

class TestPreprocessing:
    def test_compression_augmentation(self):
        from utils.preprocessing import DeepfakeAugmentor
        aug = DeepfakeAugmentor()
        img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
        out = aug.add_compression_artifacts(img, quality=40)
        assert out.size == img.size

    def test_noise_augmentation(self):
        from utils.preprocessing import DeepfakeAugmentor
        aug = DeepfakeAugmentor()
        img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
        out = aug.add_noise(img, sigma=15.0)
        assert out.size == img.size

    def test_face_swap_simulation(self):
        from utils.preprocessing import DeepfakeAugmentor
        aug = DeepfakeAugmentor()
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        out = aug.face_swap_simulation(img)
        assert out.size == img.size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
