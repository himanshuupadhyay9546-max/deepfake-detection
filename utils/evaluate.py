"""
Evaluation & Metrics
=====================
- Comprehensive metric computation
- Cross-dataset evaluation
- ROC/PR curve plotting
- Per-manipulation-type breakdown
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, classification_report,
    confusion_matrix, f1_score, accuracy_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict:
    """Compute full suite of binary classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    auc      = roc_auc_score(y_true, y_prob)
    ap       = average_precision_score(y_true, y_prob)
    f1       = f1_score(y_true, y_pred, zero_division=0)
    acc      = accuracy_score(y_true, y_pred)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    fpr      = fp / (fp + tn + 1e-8)

    return {
        "auc": round(auc, 4),
        "average_precision": round(ap, 4),
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "fpr": round(fpr, 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold that maximizes Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = np.argmax(j)
    return float(thresholds[idx])


def plot_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray, save_path: Optional[str] = None
) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#00C853", lw=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#00C853")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Deepfake Detection")
    ax.legend(loc="lower right")
    ax.set_facecolor("#0d0d0d")
    fig.patch.set_facecolor("#0d0d0d")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def plot_pr_curve(
    y_true: np.ndarray, y_prob: np.ndarray, save_path: Optional[str] = None
) -> plt.Figure:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#FF6D00", lw=2, label=f"PR (AP = {ap:.4f})")
    ax.fill_between(recall, precision, alpha=0.15, color="#FF6D00")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.set_facecolor("#0d0d0d")
    fig.patch.set_facecolor("#0d0d0d")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def plot_training_history(history: Dict, save_path: Optional[str] = None) -> plt.Figure:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0d0d0d")

    config = {"color": "white", "facecolor": "#111"}

    # Loss
    axes[0].plot(epochs, history["train_loss"], "#00C853", label="Train")
    axes[0].plot(epochs, history["val_loss"],   "#FF6D00", label="Val")
    axes[0].set_title("Loss", color="white"); axes[0].legend()

    # AUC
    axes[1].plot(epochs, history["val_auc"], "#00B0FF")
    axes[1].set_title("Validation AUC", color="white")

    # F1
    axes[2].plot(epochs, history["val_f1"], "#E040FB")
    axes[2].set_title("Validation F1", color="white")

    for ax in axes:
        ax.set_facecolor("#111")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def evaluate_on_dataset(engine, dataset, batch_size: int = 32) -> Dict:
    """Run full evaluation on a dataset and return metrics."""
    from torch.utils.data import DataLoader
    import torch

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_probs, all_labels = [], []

    engine.model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(engine.device)
            out  = engine.model(imgs)
            all_probs.extend(out["probability"].cpu().numpy())
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    threshold = find_optimal_threshold(y_true, y_prob)
    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    metrics["optimal_threshold"] = round(threshold, 4)

    report = classification_report(
        y_true, (y_prob >= threshold).astype(int),
        target_names=["Real", "Fake"], output_dict=True
    )
    metrics["classification_report"] = report
    return metrics
