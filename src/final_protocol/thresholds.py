"""Validation-only threshold selection utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.training.metrics import compute_validation_metrics


def threshold_candidates(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError("probabilities must be non-empty and one-dimensional")
    if not np.all(np.isfinite(probs)) or np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("probabilities must be finite values in [0, 1]")
    values = np.unique(np.concatenate(([0.0, 1.0], probs)))
    return np.asarray(sorted(float(value) for value in values), dtype=float)


def threshold_sort_key(row: dict[str, float]) -> tuple[float, float, float, float]:
    """Tie-break key: balanced accuracy, sensitivity, closeness to 0.5, smaller value."""

    threshold = float(row["threshold"])
    return (
        float(row["balanced_accuracy"]),
        float(row["sensitivity"]),
        -abs(threshold - 0.5),
        -threshold,
    )


def select_balanced_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, float | int]] = []
    for threshold in threshold_candidates(probabilities):
        metrics = compute_validation_metrics(y_true, probabilities, threshold=float(threshold))
        rows.append(metrics)
    frame = pd.DataFrame(rows)
    frame["threshold_distance_from_0_5"] = (frame["threshold"] - 0.5).abs()
    frame = frame.sort_values(
        by=["balanced_accuracy", "sensitivity", "threshold_distance_from_0_5", "threshold"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return float(frame.iloc[0]["threshold"]), frame
