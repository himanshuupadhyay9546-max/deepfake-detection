"""
Training Pipeline for Deepfake Detection
==========================================
Handles dataset loading, augmentation, training loops,
mixed-precision training, and checkpointing.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, classification_report

from models.detector import DeepfakeDetector, build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────

class DeepfakeDataset(Dataset):
    """
    Expects a directory layout:
        root/
          real/   ← real face images
          fake/   ← deepfake images
    OR a manifest.json:
        [{"path": "...", "label": 0|1}, ...]
    """

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 224,
        augment: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size

        self.samples: List[Dict] = self._load_samples()
        self.transform = self._build_transform(augment and split == "train")

    def _load_samples(self) -> List[Dict]:
        manifest = self.root / "manifest.json"
        if manifest.exists():
            with open(manifest) as f:
                return json.load(f)

        samples = []
        for label, cls in enumerate(["real", "fake"]):
            folder = self.root / cls
            if not folder.exists():
                continue
            for img_path in sorted(folder.glob("**/*.jpg")) + \
                            sorted(folder.glob("**/*.png")) + \
                            sorted(folder.glob("**/*.jpeg")):
                samples.append({"path": str(img_path), "label": label})

        logger.info(f"Loaded {len(samples)} samples from {self.root}")
        return samples

    def _build_transform(self, augment: bool) -> T.Compose:
        base = [
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=self.MEAN, std=self.STD),
        ]
        if not augment:
            return T.Compose(base)

        return T.Compose([
            T.Resize((int(self.image_size * 1.1), int(self.image_size * 1.1))),
            T.RandomCrop(self.image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.5),
            T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.3),
            T.RandomGrayscale(p=0.05),
            T.ToTensor(),
            T.Normalize(mean=self.MEAN, std=self.STD),
            T.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s["path"]).convert("RGB")
        return self.transform(img), torch.tensor(s["label"], dtype=torch.float32)

    def get_class_weights(self) -> torch.Tensor:
        labels = [s["label"] for s in self.samples]
        counts = np.bincount(labels)
        weights = 1.0 / counts
        sample_weights = torch.tensor([weights[l] for l in labels], dtype=torch.float32)
        return sample_weights


# ─────────────────────────────────────────
#  Loss Functions
# ─────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


import torch.nn.functional as F


# ─────────────────────────────────────────
#  Trainer
# ─────────────────────────────────────────

class Trainer:
    def __init__(self, config: dict):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.model, _ = build_model(
            pretrained=config.get("pretrained", True), device=str(self.device)
        )

        # Optimizer with layerwise LR decay
        backbone_params = list(self.model.backbone.parameters())
        head_params = (
            list(self.model.cbam.parameters())
            + list(self.model.freq_branch.parameters())
            + list(self.model.classifier.parameters())
        )
        self.optimizer = optim.AdamW([
            {"params": backbone_params, "lr": config["lr"] * 0.1},
            {"params": head_params,    "lr": config["lr"]},
        ], weight_decay=config.get("weight_decay", 1e-4))

        total_steps = config["epochs"] * config.get("steps_per_epoch", 100)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=1e-6
        )

        self.criterion = FocalLoss(
            alpha=config.get("focal_alpha", 0.25),
            gamma=config.get("focal_gamma", 2.0),
        )
        self.scaler = GradScaler()
        self.best_auc = 0.0
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [], "val_auc": [], "val_f1": []
        }

    def _run_epoch(self, loader: DataLoader, train: bool) -> Dict:
        self.model.train(train)
        total_loss = 0.0
        all_probs, all_labels = [], []

        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            with autocast():
                out = self.model(imgs)
                loss = self.criterion(out["logit"], labels)

            if train:
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

            total_loss += loss.item() * imgs.size(0)
            all_probs.extend(out["probability"].detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        n = len(loader.dataset)
        avg_loss = total_loss / n
        all_probs  = np.array(all_probs)
        all_labels = np.array(all_labels)
        auc = roc_auc_score(all_labels, all_probs) if not train else 0.0
        f1  = f1_score(all_labels, (all_probs > 0.5).astype(int), zero_division=0)

        return {"loss": avg_loss, "auc": auc, "f1": f1}

    def train(self, train_dataset, val_dataset):
        cfg = self.cfg

        # Weighted sampler for imbalanced data
        sample_weights = train_dataset.get_class_weights()
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["batch_size"],
            sampler=sampler,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["batch_size"] * 2,
            shuffle=False,
            num_workers=cfg.get("num_workers", 4),
        )

        save_dir = Path(cfg.get("save_dir", "checkpoints"))
        save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, cfg["epochs"] + 1):
            t0 = time.time()
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics   = self._run_epoch(val_loader,   train=False)

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_auc"].append(val_metrics["auc"])
            self.history["val_f1"].append(val_metrics["f1"])

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:03d}/{cfg['epochs']} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val AUC: {val_metrics['auc']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Save best model
            if val_metrics["auc"] > self.best_auc:
                self.best_auc = val_metrics["auc"]
                ckpt = save_dir / "best_model.pth"
                torch.save(self.model.state_dict(), ckpt)
                logger.info(f"  ✓ Saved best model (AUC={self.best_auc:.4f})")

            # Periodic checkpoint
            if epoch % cfg.get("save_every", 5) == 0:
                ckpt = save_dir / f"epoch_{epoch:03d}.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "history": self.history,
                    },
                    ckpt,
                )

        # Save training history
        with open(save_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        logger.info(f"\nTraining complete. Best Val AUC: {self.best_auc:.4f}")
        return self.history


def get_default_config() -> dict:
    return {
        "lr": 1e-4,
        "epochs": 30,
        "batch_size": 32,
        "weight_decay": 1e-4,
        "pretrained": True,
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "num_workers": 4,
        "save_dir": "checkpoints",
        "save_every": 5,
    }
