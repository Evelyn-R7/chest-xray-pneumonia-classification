"""Evaluate a trained model once on the complete validation set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

if __package__:
    from .data_pipeline import CLASS_MAPPING
    from .training.metrics import compute_validation_metrics
else:
    from data_pipeline import CLASS_MAPPING
    from training.metrics import compute_validation_metrics


def _save_learning_curves(history_path: Path, output_path: Path) -> None:
    history = pd.read_csv(history_path)
    candidates = [
        ("loss", "val_loss"),
        ("binary_accuracy", "val_binary_accuracy"),
        ("roc_auc", "val_roc_auc"),
        ("pr_auc", "val_pr_auc"),
    ]
    available = [(train, val) for train, val in candidates if train in history and val in history]
    if not available:
        raise ValueError(f"No plottable train/validation columns in {history_path}")
    fig, axes = plt.subplots(len(available), 1, figsize=(8, 4 * len(available)), squeeze=False)
    epochs = np.arange(1, len(history) + 1)
    for axis, (train, val) in zip(axes.flat, available):
        axis.plot(epochs, history[train], label=train)
        axis.plot(epochs, history[val], label=val)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(train)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def evaluate_validation(
    model: tf.keras.Model,
    validation_dataset: tf.data.Dataset,
    validation_frame: pd.DataFrame,
    output_dir: str | Path,
    threshold: float = 0.5,
    history_filename: str = "history.csv",
) -> dict[str, Any]:
    """Predict and evaluate only the supplied validation data at a fixed threshold."""
    output_dir = Path(output_dir)
    if "resolved_filepath" not in validation_frame:
        raise ValueError("validation_frame must contain resolved_filepath")
    y_true = validation_frame["label"].map(CLASS_MAPPING).to_numpy(dtype=int)
    probabilities = model.predict(validation_dataset, verbose=1).reshape(-1)
    if len(probabilities) != len(validation_frame):
        raise RuntimeError("Validation predictions and manifest rows have different lengths")
    metrics = compute_validation_metrics(y_true, probabilities, threshold)
    predicted = (probabilities >= threshold).astype(int)

    predictions = pd.DataFrame(
        {
            "filepath": validation_frame["resolved_filepath"].to_numpy(),
            "filename": validation_frame["filename"].to_numpy(),
            "patient_id": validation_frame["patient_id"].to_numpy(),
            "true_label": y_true,
            "predicted_probability": probabilities,
            "predicted_label": predicted,
            "threshold": float(threshold),
        }
    )
    predictions.to_csv(output_dir / "val_predictions.csv", index=False)
    (output_dir / "val_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    ConfusionMatrixDisplay.from_predictions(y_true, predicted, labels=[0, 1], cmap="Blues")
    plt.title("Validation confusion matrix (threshold=0.5)")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_val.png", dpi=160)
    plt.close()

    RocCurveDisplay.from_predictions(y_true, probabilities)
    plt.title("Validation ROC curve")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve_val.png", dpi=160)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, probabilities)
    plt.title("Validation precision-recall curve")
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve_val.png", dpi=160)
    plt.close()
    _save_learning_curves(output_dir / history_filename, output_dir / "learning_curves.png")
    return metrics
