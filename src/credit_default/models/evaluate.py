"""Метрики бинарной классификации и графики качества"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def classification_metrics(
    y_true,
    y_prob,
    threshold: float = 0.5,
) -> dict[str, float]:
    """ROC-AUC, precision, recall и F1 при заданном пороге"""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def save_roc_curve(y_true, y_prob, path: Path, title: str) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = float(roc_auc_score(y_true, y_prob))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1d4e89", lw=2, label=f"ROC-AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="Случайная модель")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return auc


def save_confusion_matrix(y_true, y_pred, path: Path, title: str) -> None:
    matrix = confusion_matrix(np.asarray(y_true).astype(int), np.asarray(y_pred))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_xticks([0, 1], ["Нет дефолта", "Дефолт"])
    ax.set_yticks([0, 1], ["Нет дефолта", "Дефолт"])
    ax.set_xlabel("Предсказание")
    ax.set_ylabel("Факт")
    ax.set_title(title)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_feature_importance(
    names, importances, path: Path, title: str, top_n: int = 15
) -> None:
    order = np.argsort(importances)[::-1][:top_n]
    selected_names = [names[index] for index in order][::-1]
    selected_values = importances[order][::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(selected_names, selected_values, color="#3c6e71")
    ax.set_xlabel("Падение ROC-AUC при перемешивании признака")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
