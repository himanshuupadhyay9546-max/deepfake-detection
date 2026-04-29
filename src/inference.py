"""
Inference & Explainability
===========================
- Single image / batch prediction
- Video analysis (frame sampling)
- Grad-CAM heatmap generation
- Face detection + cropping
"""

import io
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from models.detector import DeepfakeDetector


# ─────────────────────────────────────────
#  Transforms
# ─────────────────────────────────────────

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def get_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Reverse normalization for visualization."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


# ─────────────────────────────────────────
#  Result Dataclass
# ─────────────────────────────────────────

@dataclass
class DetectionResult:
    label: str               # "REAL" or "FAKE"
    probability: float       # P(fake)
    confidence: float        # certainty score 0-1
    heatmap: Optional[np.ndarray] = None
    face_bbox: Optional[Tuple] = None
    frame_results: List[Dict] = field(default_factory=list)  # for video

    @property
    def is_fake(self) -> bool:
        return self.label == "FAKE"

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "probability": round(self.probability, 4),
            "confidence": round(self.confidence, 4),
            "is_fake": self.is_fake,
        }


# ─────────────────────────────────────────
#  GradCAM
# ─────────────────────────────────────────

class GradCAM:
    """Gradient-weighted Class Activation Mapping."""

    def __init__(self, model: DeepfakeDetector, target_layer: torch.nn.Module):
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x: torch.Tensor, class_idx: int = 0) -> np.ndarray:
        self.model.zero_grad()
        out = self.model(x)
        score = out["logit"][0]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def overlay_heatmap(
        self, img_np: np.ndarray, cam: np.ndarray, alpha: float = 0.45
    ) -> np.ndarray:
        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (alpha * heatmap + (1 - alpha) * img_np).astype(np.uint8)
        return overlay


# ─────────────────────────────────────────
#  Face Detector (lightweight Haar cascade)
# ─────────────────────────────────────────

class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)

    def detect(self, img_np: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        # Add margin
        margin = int(0.2 * min(w, h))
        ih, iw = img_np.shape[:2]
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(iw, x + w + margin)
        y2 = min(ih, y + h + margin)
        return (x1, y1, x2, y2)

    def crop_face(self, img_np: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple]]:
        bbox = self.detect(img_np)
        if bbox is None:
            return img_np, None
        x1, y1, x2, y2 = bbox
        return img_np[y1:y2, x1:x2], bbox


# ─────────────────────────────────────────
#  Main Inference Engine
# ─────────────────────────────────────────

class DeepfakeInference:
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        image_size: int = 224,
        generate_heatmap: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.generate_heatmap = generate_heatmap

        self.model = DeepfakeDetector(pretrained=model_path is None)
        if model_path and Path(model_path).exists():
            state = torch.load(model_path, map_location=self.device)
            if "model_state" in state:
                state = state["model_state"]
            self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        self.transform = get_transform(image_size)
        self.face_detector = FaceDetector()

        if generate_heatmap:
            target_layer = self.model.get_gradcam_target_layer()
            self.gradcam = GradCAM(self.model, target_layer)
        else:
            self.gradcam = None

    @torch.no_grad()
    def _predict_tensor(self, x: torch.Tensor) -> Dict:
        if self.generate_heatmap and self.gradcam:
            x.requires_grad_(False)
            with torch.enable_grad():
                out = self.model(x)
        else:
            out = self.model(x)
        return {k: v.cpu() for k, v in out.items()}

    def predict_image(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        crop_face: bool = True,
    ) -> DetectionResult:
        # Load image
        if isinstance(image, (str, Path)):
            img_pil = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img_pil = Image.fromarray(image)
        else:
            img_pil = image

        img_np = np.array(img_pil)

        # Optional face crop
        face_bbox = None
        if crop_face:
            cropped, face_bbox = self.face_detector.crop_face(img_np)
            if face_bbox is not None:
                img_pil = Image.fromarray(cropped)

        # Preprocess
        tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        # Heatmap
        heatmap_overlay = None
        if self.generate_heatmap and self.gradcam:
            tensor.requires_grad_(True)
            out = self.model(tensor)
            logit = out["logit"]
            logit.backward()
            cam = self.gradcam.generate(tensor)
            img_resized = np.array(img_pil.resize((self.image_size, self.image_size)))
            heatmap_overlay = self.gradcam.overlay_heatmap(img_resized, cam)
        else:
            with torch.no_grad():
                out = self.model(tensor)

        prob = out["probability"].item()
        conf = abs(prob - 0.5) * 2

        return DetectionResult(
            label="FAKE" if prob > 0.5 else "REAL",
            probability=prob,
            confidence=conf,
            heatmap=heatmap_overlay,
            face_bbox=face_bbox,
        )

    def predict_video(
        self,
        video_path: str,
        frame_interval: int = 10,
        max_frames: int = 50,
        crop_face: bool = True,
    ) -> DetectionResult:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frame_results = []
        frame_count = 0
        processed = 0

        while processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.predict_image(frame_rgb, crop_face=crop_face)
                frame_results.append({
                    "frame": frame_count,
                    "probability": result.probability,
                    "label": result.label,
                })
                processed += 1
            frame_count += 1

        cap.release()

        if not frame_results:
            raise ValueError("No frames could be processed.")

        probs = [r["probability"] for r in frame_results]
        avg_prob = float(np.mean(probs))
        conf = float(abs(avg_prob - 0.5) * 2)

        return DetectionResult(
            label="FAKE" if avg_prob > 0.5 else "REAL",
            probability=avg_prob,
            confidence=conf,
            frame_results=frame_results,
        )

    def predict_batch(
        self, images: List[Union[str, np.ndarray]], crop_face: bool = True
    ) -> List[DetectionResult]:
        return [self.predict_image(img, crop_face=crop_face) for img in images]
