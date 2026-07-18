"""Full-validation binary classification metrics at a fixed threshold."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def compute_validation_metrics(
    true_labels: Iterable[int],
    probabilities: Iterable[float],
    threshold: float = 0.5,
) -> dict[str, float | int]:
    y_true = np.asarray(list(true_labels), dtype=int)
    y_prob = np.asarray(list(probabilities), dtype=float)
    if y_true.ndim != 1 or y_prob.ndim != 1 or y_true.size != y_prob.size or not y_true.size:
        raise ValueError("Labels and probabilities must be non-empty one-dimensional arrays of equal length")
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("true_labels must contain only binary labels 0 and 1")
    if not np.all(np.isfinite(y_prob)) or np.any((y_prob < 0.0) | (y_prob > 1.0)):
        raise ValueError("probabilities must be finite values in [0, 1]")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both_classes = np.unique(y_true).size == 2
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": _safe_divide(tn, tn + fp),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if has_both_classes else 0.0,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if has_both_classes else 0.0,
        "npv": _safe_divide(tn, tn + fn),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
