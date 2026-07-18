"""Callback construction for the CNN baseline protocol."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf


def build_callbacks(output_dir: str | Path, config: dict) -> list[tf.keras.callbacks.Callback]:
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")
    monitor = str(config.get("monitor", "val_loss"))
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=output_dir / "best_model.keras",
            monitor=monitor,
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=int(config.get("early_stopping_patience", 5)),
            min_delta=0.0001,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=float(config.get("reduce_lr_factor", 0.5)),
            patience=int(config.get("reduce_lr_patience", 2)),
            min_lr=float(config.get("minimum_learning_rate", 1e-6)),
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(output_dir / "history.csv"),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
