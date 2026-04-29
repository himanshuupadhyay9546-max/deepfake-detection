"""
Data Preprocessing & Augmentation Utilities
=============================================
- Video frame extraction
- Dataset splitting
- Custom augmentation policies
- Data validation
"""

import os
import json
import random
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm


# ─────────────────────────────────────────
#  Video Frame Extractor
# ─────────────────────────────────────────

def extract_frames(
    video_path: str,
    output_dir: str,
    frame_interval: int = 5,
    max_frames: Optional[int] = None,
    face_only: bool = True,
    min_face_confidence: float = 0.8,
) -> List[str]:
    """
    Extract frames from video at specified interval.
    Optionally crops to detected face regions.
    Returns list of saved frame paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    face_cascade = None
    if face_only:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)

    saved_paths = []
    frame_idx = 0
    saved_count = 0
    video_name = Path(video_path).stem

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            if max_frames and saved_count >= max_frames:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if face_only and face_cascade is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    margin = int(0.25 * min(w, h))
                    ih, iw = frame_rgb.shape[:2]
                    x1, y1 = max(0, x - margin), max(0, y - margin)
                    x2, y2 = min(iw, x + w + margin), min(ih, y + h + margin)
                    frame_rgb = frame_rgb[y1:y2, x1:x2]
                else:
                    frame_idx += 1
                    continue

            out_path = output_dir / f"{video_name}_frame{frame_idx:06d}.jpg"
            Image.fromarray(frame_rgb).save(str(out_path), quality=95)
            saved_paths.append(str(out_path))
            saved_count += 1

        frame_idx += 1

    cap.release()
    return saved_paths


# ─────────────────────────────────────────
#  Dataset Preparation
# ─────────────────────────────────────────

def build_manifest(
    data_dir: str,
    val_split: float = 0.15,
    test_split: float = 0.10,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    Scan data_dir for real/ and fake/ subdirectories,
    build stratified train/val/test manifests.
    """
    random.seed(seed)
    samples = {"real": [], "fake": []}

    for label, cls in enumerate(["real", "fake"]):
        folder = Path(data_dir) / cls
        if not folder.exists():
            print(f"Warning: {folder} not found, skipping.")
            continue
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() in exts:
                samples[cls].append({"path": str(p), "label": label})

    splits = {}
    for cls, cls_samples in samples.items():
        random.shuffle(cls_samples)
        n = len(cls_samples)
        n_val  = int(n * val_split)
        n_test = int(n * test_split)
        n_train = n - n_val - n_test

        splits.setdefault("train", []).extend(cls_samples[:n_train])
        splits.setdefault("val",   []).extend(cls_samples[n_train:n_train + n_val])
        splits.setdefault("test",  []).extend(cls_samples[n_train + n_val:])

    for split, data in splits.items():
        random.shuffle(data)
        out_path = Path(data_dir) / f"manifest_{split}.json"
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  {split}: {len(data)} samples → {out_path}")

    return splits


def validate_dataset(data_dir: str) -> Dict:
    """Check dataset integrity: count images, find corrupt files."""
    data_dir = Path(data_dir)
    report = {"total": 0, "real": 0, "fake": 0, "corrupt": 0, "sizes": []}

    for cls in ["real", "fake"]:
        folder = data_dir / cls
        if not folder.exists():
            continue
        for p in folder.rglob("*.jpg"):
            report["total"] += 1
            report[cls] += 1
            try:
                img = Image.open(p)
                img.verify()
                report["sizes"].append(img.size if hasattr(img, "size") else (0, 0))
            except Exception:
                report["corrupt"] += 1

    if report["sizes"]:
        sizes = np.array(report["sizes"])
        report["avg_width"]  = float(sizes[:, 0].mean())
        report["avg_height"] = float(sizes[:, 1].mean())

    return report


# ─────────────────────────────────────────
#  Custom Augmentation Policies
# ─────────────────────────────────────────

class DeepfakeAugmentor:
    """
    Additional augmentations that mimic deepfake artifacts,
    useful for hard-negative mining.
    """

    @staticmethod
    def add_compression_artifacts(img: Image.Image, quality: int = 50) -> Image.Image:
        buf = __import__("io").BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    @staticmethod
    def add_blur(img: Image.Image, radius: float = 1.5) -> Image.Image:
        return img.filter(ImageFilter.GaussianBlur(radius))

    @staticmethod
    def add_noise(img: Image.Image, sigma: float = 10.0) -> Image.Image:
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, sigma, arr.shape)
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

    @staticmethod
    def face_swap_simulation(img: Image.Image) -> Image.Image:
        """Simulate blending boundary artifacts around face region."""
        arr = np.array(img)
        h, w = arr.shape[:2]
        cx, cy = w // 2, h // 2
        r = min(w, h) // 3

        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (cx, cy), r, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (51, 51), 20)

        blurred = cv2.GaussianBlur(arr, (15, 15), 5)
        mask3 = mask[:, :, np.newaxis]
        result = (arr * mask3 + blurred * (1 - mask3)).astype(np.uint8)
        return Image.fromarray(result)

    def apply_random(self, img: Image.Image, p: float = 0.3) -> Image.Image:
        augments = [
            self.add_compression_artifacts,
            self.add_blur,
            self.add_noise,
            self.face_swap_simulation,
        ]
        for aug in augments:
            if random.random() < p:
                img = aug(img)
        return img
