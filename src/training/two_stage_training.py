"""Strict two-stage transfer-learning orchestration."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import tensorflow as tf


def build_loss(config: dict[str, Any]) -> tf.keras.losses.Loss:
    loss_name = config.get("loss", "binary_crossentropy")
    if loss_name == "binary_crossentropy":
        return tf.keras.losses.BinaryCrossentropy()
    if loss_name == "binary_focal_crossentropy":
        focal = config.get("focal_loss") or {}
        return tf.keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=bool(focal.get("apply_class_balancing", False)),
            alpha=float(focal["alpha"]),
            gamma=float(focal.get("gamma", 2.0)),
            from_logits=bool(focal.get("from_logits", False)),
            label_smoothing=float(focal.get("label_smoothing", 0.0)),
        )
    raise ValueError(f"Unsupported loss: {loss_name}")


def resolve_class_weight(config: dict[str, Any]) -> dict[int, float] | None:
    weights = config.get("class_weight")
    if weights is None:
        return None
    return {int(key): float(value) for key, value in weights.items()}


def compile_binary_model(model: tf.keras.Model, learning_rate: float, config: dict[str, Any] | None = None) -> None:
    """Compile after every trainable-state transition."""
    config = config or {"loss": "binary_crossentropy"}
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=build_loss(config),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )


def build_phase_callbacks(
    output_dir: str | Path,
    phase: str,
    early_stopping_patience: int,
    reduce_lr_patience: int,
    reduce_lr_factor: float,
    minimum_learning_rate: float,
) -> list[tf.keras.callbacks.Callback]:
    output_dir = Path(output_dir)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            output_dir / f"{phase}_best.keras", monitor="val_loss",
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=early_stopping_patience, min_delta=0.0001,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=reduce_lr_patience, factor=reduce_lr_factor,
            min_lr=minimum_learning_rate, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(output_dir / f"{phase}_history.csv"),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def resolve_stage_epochs(config: dict[str, Any], run_type: str) -> tuple[int, int]:
    if run_type == "pilot":
        return 1, 1
    if run_type == "full":
        return int(config["head_max_epochs"]), int(config["fine_tune_max_epochs"])
    raise ValueError("run_type must be pilot or full")


def _validate_history(history: tf.keras.callbacks.History, phase: str) -> None:
    values = [value for sequence in history.history.values() for value in sequence]
    if not values or not np.all(np.isfinite(np.asarray(values, dtype=float))):
        raise FloatingPointError(f"{phase} history contains NaN or Inf")


def run_two_stage_training(
    model: tf.keras.Model,
    train_dataset,
    validation_dataset,
    output_dir: str | Path,
    config: dict[str, Any],
    run_type: str,
    freeze_fn: Callable[[tf.keras.Model], list[str]],
    unfreeze_fn: Callable[[tf.keras.Model], list[str]],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    phase1_epochs, phase2_epochs = resolve_stage_epochs(config, run_type)
    class_weight = resolve_class_weight(config)

    freeze_fn(model)
    compile_binary_model(model, float(config["head_learning_rate"]), config)
    phase1_callbacks = build_phase_callbacks(
        output_dir, "phase1", int(config["head_early_stopping_patience"]),
        int(config["reduce_lr_patience"]), float(config["reduce_lr_factor"]),
        float(config["minimum_learning_rate"]),
    )
    try:
        phase1_history = model.fit(
            train_dataset, validation_data=validation_dataset, epochs=phase1_epochs,
            callbacks=phase1_callbacks, class_weight=class_weight, verbose=2,
        )
    except tf.errors.ResourceExhaustedError as exc:
        raise RuntimeError("GPU OOM during phase1; batch size was not changed") from exc
    _validate_history(phase1_history, "phase1")

    phase1_path = output_dir / "phase1_best.keras"
    if not phase1_path.is_file():
        raise RuntimeError("phase1 checkpoint was not created")
    model = tf.keras.models.load_model(phase1_path)
    unfrozen_layers = unfreeze_fn(model)
    compile_binary_model(model, float(config["fine_tune_learning_rate"]), config)
    phase2_callbacks = build_phase_callbacks(
        output_dir, "phase2", int(config["fine_tune_early_stopping_patience"]),
        int(config["reduce_lr_patience"]), float(config["reduce_lr_factor"]),
        float(config["minimum_learning_rate"]),
    )
    initial_epoch = len(phase1_history.epoch)
    try:
        phase2_history = model.fit(
            train_dataset, validation_data=validation_dataset,
            initial_epoch=initial_epoch, epochs=initial_epoch + phase2_epochs,
            callbacks=phase2_callbacks, class_weight=class_weight, verbose=2,
        )
    except tf.errors.ResourceExhaustedError as exc:
        raise RuntimeError("GPU OOM during phase2; batch size was not changed") from exc
    _validate_history(phase2_history, "phase2")

    phase2_path = output_dir / "phase2_best.keras"
    if not phase2_path.is_file():
        raise RuntimeError("phase2 checkpoint was not created")
    phase1_frame = pd.read_csv(output_dir / "phase1_history.csv")
    phase2_frame = pd.read_csv(output_dir / "phase2_history.csv")
    combined = pd.concat([phase1_frame, phase2_frame], ignore_index=True)
    combined.to_csv(output_dir / "combined_history.csv", index=False)
    phase1_min = float(phase1_frame["val_loss"].min())
    phase2_min = float(phase2_frame["val_loss"].min())
    best_phase = "phase1" if phase1_min <= phase2_min else "phase2"
    best_source = phase1_path if best_phase == "phase1" else phase2_path
    shutil.copy2(best_source, output_dir / "best_model.keras")
    best_model = tf.keras.models.load_model(output_dir / "best_model.keras")
    return {
        "model": best_model,
        "best_phase": best_phase,
        "phase1_min_val_loss": phase1_min,
        "phase2_min_val_loss": phase2_min,
        "phase1_epochs": len(phase1_frame),
        "phase2_epochs": len(phase2_frame),
        "unfrozen_layers": unfrozen_layers,
        "phase1_callbacks": phase1_callbacks,
        "phase2_callbacks": phase2_callbacks,
    }
