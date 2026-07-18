"""Reporting and plotting helpers for final test evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from src.final_protocol.calibration import expected_calibration_error
from src.final_test.safety import sha256_file


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_standard_plots(y_true: np.ndarray, probabilities: np.ndarray, output_dir: str | Path, threshold_0_5: float, balanced_threshold: float) -> None:
    output = Path(output_dir)
    for threshold, filename, title in [
        (threshold_0_5, "confusion_matrix_threshold_0_5.png", "Test confusion matrix (threshold=0.5)"),
        (balanced_threshold, "confusion_matrix_balanced_threshold.png", "Test confusion matrix (balanced threshold)"),
    ]:
        predicted = (probabilities >= threshold).astype(int)
        ConfusionMatrixDisplay.from_predictions(y_true, predicted, labels=[0, 1], cmap="Blues")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output / filename, dpi=160)
        plt.close()
    RocCurveDisplay.from_predictions(y_true, probabilities)
    plt.title("Final test ROC curve")
    plt.tight_layout()
    plt.savefig(output / "roc_curve_test.png", dpi=160)
    plt.close()
    PrecisionRecallDisplay.from_predictions(y_true, probabilities)
    plt.title("Final test precision-recall curve")
    plt.tight_layout()
    plt.savefig(output / "pr_curve_test.png", dpi=160)
    plt.close()
    save_calibration_curve(y_true, probabilities, output / "calibration_curve_test.png")


def save_calibration_curve(y_true: np.ndarray, probabilities: np.ndarray, output_path: str | Path) -> None:
    bins = np.linspace(0, 1, 16)
    xs, ys, sizes = [], [], []
    for index in range(15):
        left, right = bins[index], bins[index + 1]
        mask = (probabilities >= left) & ((probabilities <= right) if index == 14 else (probabilities < right))
        if np.any(mask):
            xs.append(float(np.mean(probabilities[mask])))
            ys.append(float(np.mean(y_true[mask])))
            sizes.append(max(30, int(mask.sum()) * 2))
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    plt.scatter(xs, ys, s=sizes, alpha=0.75)
    plt.title(f"Final test calibration curve; ECE={expected_calibration_error(y_true, probabilities):.4f}")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive fraction")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_bootstrap_intervals(ci_payload: dict[str, Any], output_path: str | Path) -> None:
    rows = []
    for point, metrics in ci_payload["intervals"].items():
        for metric, interval in metrics.items():
            rows.append({"operating_point": point, "metric": metric, **interval})
    frame = pd.DataFrame(rows)
    plt.figure(figsize=(10, max(4, len(frame) * 0.22)))
    y = np.arange(len(frame))
    center = (frame["ci_lower"] + frame["ci_upper"]) / 2
    error = np.vstack([center - frame["ci_lower"], frame["ci_upper"] - center])
    plt.errorbar(center, y, xerr=error, fmt="o")
    plt.yticks(y, frame["operating_point"] + " / " + frame["metric"])
    plt.xlabel("Bootstrap 95% CI")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_final_test_report(
    report_path: str | Path,
    evaluation_record: dict[str, Any],
    point_estimates: dict[str, Any],
    ci_payload: dict[str, Any],
    patient_metrics: dict[str, Any],
    generalization_gap: dict[str, Any],
    anomalies: list[str],
) -> None:
    text = f"""# Final test results

This report is generated only after the frozen protocol has been evaluated once.

- Frozen protocol SHA-256: `{evaluation_record['frozen_protocol_sha256']}`
- Test manifest SHA-256: `{evaluation_record['test_manifest_sha256']}`
- Ensemble: EfficientNetB0 control seeds 42, 2025, 2026 with equal probability averaging
- Calibration: none
- Primary balanced threshold: `{evaluation_record['balanced_threshold']}`
- Benchmark threshold: `{evaluation_record['benchmark_threshold']}`

## Point estimates

```json
{json.dumps(point_estimates, indent=2)}
```

## Patient-cluster bootstrap 95% CI

```json
{json.dumps(ci_payload, indent=2)}
```

## Secondary patient-level analysis

Patient-level probabilities average all images for each inferred patient_id. No patient-level threshold was re-selected.

```json
{json.dumps(patient_metrics, indent=2)}
```

## Validation-test generalization gap

These are descriptive test metric minus frozen validation metric comparisons only. They are not used to alter the model, thresholds, calibration, or training.

```json
{json.dumps(generalization_gap, indent=2)}
```

## Anomalies

{json.dumps(anomalies, indent=2)}

No retraining, recalibration, threshold adjustment, or ensemble reweighting was performed after seeing test results.

Limitations: patient_id values are inferred from filenames, the data come from a single public source, no external validation is included, and this model is not clinically deployable or a replacement for clinician judgment.
"""
    Path(report_path).write_text(text, encoding="utf-8")


def file_hashes(paths: dict[str, str | Path]) -> dict[str, str]:
    return {key: sha256_file(path) for key, path in paths.items()}
