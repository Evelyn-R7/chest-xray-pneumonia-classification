"""Patient-cluster bootstrap utilities for final test metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.final_protocol.calibration import calibration_metrics
from src.final_protocol.thresholds import select_balanced_threshold  # noqa: F401 - documents no re-selection use
from src.training.metrics import compute_validation_metrics

CI_METRICS = (
    "accuracy", "precision", "sensitivity", "specificity", "f1", "balanced_accuracy",
    "npv", "roc_auc", "pr_auc", "brier_score", "log_loss", "ece",
)


def validate_patient_labels(frame: pd.DataFrame) -> None:
    label_counts = frame.groupby("patient_id")["true_label"].nunique()
    bad = label_counts[label_counts > 1]
    if not bad.empty:
        raise ValueError(f"patient_id has multiple true_label values: {bad.index.tolist()[:5]}")


def patient_level_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    validate_patient_labels(frame)
    grouped = frame.groupby("patient_id", sort=False)
    return grouped.agg(
        true_label=("true_label", "first"),
        final_probability=("final_probability", "mean"),
        image_count=("filename", "size"),
    ).reset_index()


def metrics_for_probabilities(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    metrics = compute_validation_metrics(y_true, probabilities, threshold)
    metrics.update(calibration_metrics(y_true, probabilities))
    return metrics


def metrics_for_two_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    benchmark_threshold: float,
    balanced_threshold: float,
) -> dict[str, dict[str, float | int]]:
    independent = calibration_metrics(y_true, probabilities)
    benchmark = compute_validation_metrics(y_true, probabilities, benchmark_threshold)
    balanced = compute_validation_metrics(y_true, probabilities, balanced_threshold)
    for target in (benchmark, balanced):
        target.update(independent)
    return {
        "threshold_0_5": benchmark,
        "balanced_threshold": balanced,
    }


def stratified_patient_bootstrap_indices(
    frame: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    validate_patient_labels(frame)
    patient_table = frame.groupby("patient_id", sort=False)["true_label"].first().reset_index()
    sampled_patients: list[str] = []
    for label in (0, 1):
        patients = patient_table.loc[patient_table["true_label"].astype(int) == label, "patient_id"].to_numpy()
        if patients.size == 0:
            raise ValueError("Both classes must be present for stratified bootstrap")
        sampled_patients.extend(rng.choice(patients, size=patients.size, replace=True).tolist())
    chunks = []
    for patient in sampled_patients:
        chunks.append(frame.index[frame["patient_id"] == patient].to_numpy())
    return np.concatenate(chunks)


def percentile_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Finite one-dimensional values are required")
    alpha = (1.0 - confidence) / 2.0
    low, high = np.percentile(array, [100 * alpha, 100 * (1 - alpha)])
    return float(low), float(high)


def cluster_bootstrap(
    frame: pd.DataFrame,
    benchmark_threshold: float,
    balanced_threshold: float,
    replicates: int = 5000,
    seed: int = 20260718,
) -> tuple[pd.DataFrame, dict]:
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    required = {"filename", "patient_id", "true_label", "final_probability"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing bootstrap columns: {sorted(missing)}")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    failed = 0
    for replicate in range(1, replicates + 1):
        try:
            indices = stratified_patient_bootstrap_indices(frame, rng)
            sample = frame.loc[indices]
            labels = sample["true_label"].to_numpy(int)
            probs = sample["final_probability"].to_numpy(float)
            if set(np.unique(labels)) != {0, 1}:
                raise ValueError("Bootstrap replicate does not contain both classes")
            metrics = metrics_for_two_thresholds(labels, probs, benchmark_threshold, balanced_threshold)
            for scope, values in metrics.items():
                row = {"replicate": replicate, "operating_point": scope}
                row.update(values)
                rows.append(row)
        except Exception:
            failed += 1
    table = pd.DataFrame(rows)
    successful = int(table["replicate"].nunique()) if not table.empty else 0
    if failed or successful != replicates:
        raise ValueError(f"Bootstrap failed: successful={successful}, failed={failed}, expected={replicates}")
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for operating_point, group in table.groupby("operating_point"):
        summary[operating_point] = {}
        for metric in CI_METRICS:
            low, high = percentile_interval(group[metric].to_numpy(float))
            summary[operating_point][metric] = {"ci_lower": low, "ci_upper": high}
    return table, {
        "replicates_requested": replicates,
        "successful_replicates": successful,
        "failed_replicates": failed,
        "confidence_level": 0.95,
        "method": "patient-level stratified cluster bootstrap",
        "intervals": summary,
    }
