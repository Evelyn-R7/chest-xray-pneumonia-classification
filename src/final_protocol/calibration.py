"""Validation-only probability calibration utilities.

ECE definition used by this module:
Split [0, 1] into 15 equal-width bins. For every non-empty bin, compute
|mean(predicted_probability) - mean(true_label)| and weight it by the fraction
of samples in the bin. ECE is the weighted sum across bins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

CalibrationMethod = Literal["none", "platt", "isotonic"]
METHOD_SIMPLICITY_ORDER: tuple[CalibrationMethod, ...] = ("none", "platt", "isotonic")


@dataclass(frozen=True)
class CalibrationCandidate:
    """OOF predictions and metrics for one calibration method."""

    method: CalibrationMethod
    predictions: np.ndarray
    metrics: dict[str, float]
    fold_metrics: list[dict[str, float | int | str]]
    rejected_reason: str | None = None


def validate_probability_array(probabilities: np.ndarray, name: str = "probabilities") -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{name} must contain only finite values in [0, 1]")
    return values


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute equal-width-bin ECE for binary probabilities."""

    labels = np.asarray(y_true, dtype=int)
    probs = validate_probability_array(probabilities)
    if labels.shape != probs.shape:
        raise ValueError("y_true and probabilities must have the same shape")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        left = edges[index]
        right = edges[index + 1]
        if index == n_bins - 1:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        if not np.any(mask):
            continue
        confidence = float(np.mean(probs[mask]))
        accuracy = float(np.mean(labels[mask]))
        ece += float(np.mean(mask)) * abs(confidence - accuracy)
    return float(ece)


def calibration_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(y_true, dtype=int)
    probs = validate_probability_array(probabilities)
    if labels.shape != probs.shape:
        raise ValueError("y_true and probabilities must have the same shape")
    clipped = np.clip(probs, 1e-7, 1.0 - 1e-7)
    return {
        "brier_score": float(brier_score_loss(labels, probs)),
        "log_loss": float(log_loss(labels, clipped, labels=[0, 1])),
        "ece": expected_calibration_error(labels, probs, n_bins=15),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "pr_auc": float(average_precision_score(labels, probs)),
    }


def fit_calibrator(method: CalibrationMethod, train_probabilities: np.ndarray, y_train: np.ndarray):
    probs = validate_probability_array(train_probabilities)
    labels = np.asarray(y_train, dtype=int)
    if method == "none":
        return None
    if method == "platt":
        model = LogisticRegression(random_state=42, solver="lbfgs")
        model.fit(probs.reshape(-1, 1), labels)
        return model
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(probs, labels)
        return model
    raise ValueError(f"Unknown calibration method: {method}")


def apply_calibrator(method: CalibrationMethod, calibrator, probabilities: np.ndarray) -> np.ndarray:
    probs = validate_probability_array(probabilities)
    if method == "none":
        output = probs.copy()
    elif method == "platt":
        output = calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
    elif method == "isotonic":
        output = calibrator.predict(probs)
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    return validate_probability_array(np.asarray(output, dtype=float), f"{method} calibrated probabilities")


def build_oof_calibration_candidates(
    y_true: np.ndarray,
    raw_probabilities: np.ndarray,
    patient_ids: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    """Build full out-of-fold predictions and metrics for none/Platt/isotonic."""

    labels = np.asarray(y_true, dtype=int)
    raw = validate_probability_array(raw_probabilities, "raw_probabilities")
    groups = np.asarray(patient_ids)
    if labels.shape != raw.shape or labels.shape != groups.shape:
        raise ValueError("labels, probabilities, and patient_ids must have matching shape")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    methods: tuple[CalibrationMethod, ...] = ("none", "platt", "isotonic")
    oof = {method: np.full(raw.shape, np.nan, dtype=float) for method in methods}
    rows: list[dict[str, float | int | str]] = []
    assigned = np.zeros(raw.shape, dtype=bool)

    for fold, (train_idx, val_idx) in enumerate(splitter.split(raw, labels, groups), start=1):
        if set(groups[train_idx]).intersection(set(groups[val_idx])):
            raise ValueError(f"Patient group leakage detected in fold {fold}")
        assigned[val_idx] = True
        for method in methods:
            calibrator = fit_calibrator(method, raw[train_idx], labels[train_idx])
            fold_probs = apply_calibrator(method, calibrator, raw[val_idx])
            oof[method][val_idx] = fold_probs
            row = {
                "method": method,
                "fold": fold,
                "n_train": int(train_idx.size),
                "n_validation": int(val_idx.size),
            }
            row.update(calibration_metrics(labels[val_idx], fold_probs))
            rows.append(row)

    if not np.all(assigned):
        raise ValueError("Some samples did not receive OOF predictions")

    prediction_frame = pd.DataFrame({"true_label": labels, "patient_id": groups, "raw_probability": raw})
    summary: dict[str, dict[str, float]] = {}
    for method in methods:
        prediction_frame[f"oof_probability_{method}"] = validate_probability_array(oof[method], method)
        summary[method] = calibration_metrics(labels, oof[method])
    return prediction_frame, pd.DataFrame(rows), summary


def choose_calibration_method(
    summary: dict[str, dict[str, float]],
    tolerance: float = 0.001,
) -> CalibrationMethod:
    """Choose by lowest Brier; within tolerance choose simpler method."""

    for method, metrics in summary.items():
        probs = [float(value) for value in metrics.values()]
        if not np.all(np.isfinite(probs)):
            raise ValueError(f"Calibration method {method} has non-finite metrics")
    best_brier = min(float(metrics["brier_score"]) for metrics in summary.values())
    candidates = {
        method for method, metrics in summary.items()
        if float(metrics["brier_score"]) - best_brier <= tolerance
    }
    for method in METHOD_SIMPLICITY_ORDER:
        if method in candidates:
            return method
    raise ValueError("No valid calibration method candidates")
