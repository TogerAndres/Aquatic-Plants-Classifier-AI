"""Métricas agregadas reutilizables en reportes de evaluación y K-Fold."""

from __future__ import annotations

import numpy as np


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[int, float]:
    accs = {}
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() == 0:
            accs[c] = float("nan")
            continue
        accs[c] = float(np.mean(y_pred[mask] == c))
    return accs


def mean_confidence_by_correctness(probs: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    confidences = probs[np.arange(len(probs)), y_pred]
    correct_mask = y_true == y_pred
    return {
        "mean_confidence_correct": float(confidences[correct_mask].mean()) if correct_mask.any() else float("nan"),
        "mean_confidence_incorrect": float(confidences[~correct_mask].mean()) if (~correct_mask).any() else float("nan"),
    }
