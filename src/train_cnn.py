"""Train cnn_baseline_v1 using train/validation only; the test manifest is never loaded."""

from __future__ import annotations

import argparse
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tensorflow as tf
import yaml

if __package__:
    from .data_pipeline import build_dataset, load_manifest, validate_manifest
    from .evaluate_validation import evaluate_validation
    from .models.cnn_baseline import build_cnn_baseline
    from .training.callbacks import build_callbacks
    from .training.reproducibility import configure_reproducibility
else:
    from data_pipeline import build_dataset, load_manifest, validate_manifest
    from evaluate_validation import evaluate_validation
    from models.cnn_baseline import build_cnn_baseline
    from training.callbacks import build_callbacks
    from training.reproducibility import configure_reproducibility

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CONFIG_KEYS = {
    "experiment_name", "data_config", "seed", "image_size", "batch_size", "max_epochs",
    "learning_rate", "dropout_rate", "loss", "class_weight", "mixed_precision", "monitor",
    "threshold",
}


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.resolve().open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Experiment config must be a YAML mapping")
    missing = REQUIRED_CONFIG_KEYS - set(config)
    if missing:
        raise ValueError(f"Experiment config missing keys: {sorted(missing)}")
    if config["experiment_name"] != "cnn_baseline_v1":
        raise ValueError("experiment_name must be cnn_baseline_v1")
    if config["class_weight"] is not None:
        raise ValueError("Stage 4A requires class_weight: null")
    if bool(config["mixed_precision"]):
        raise ValueError("Stage 4A does not permit mixed precision")
    if config["loss"] != "binary_crossentropy":
        raise ValueError("Stage 4A loss must be binary_crossentropy")
    if float(config["threshold"]) != 0.5:
        raise ValueError("Stage 4A threshold is fixed at 0.5")
    return config


def apply_cli_overrides(
    config: dict[str, Any], max_epochs: int | None = None, seed: int | None = None
) -> dict[str, Any]:
    resolved = dict(config)
    if max_epochs is not None:
        if max_epochs < 1:
            raise ValueError("--max-epochs must be at least 1")
        resolved["max_epochs"] = int(max_epochs)
    if seed is not None:
        if seed < 0:
            raise ValueError("--seed must be non-negative")
        resolved["seed"] = int(seed)
    return resolved


def create_run_directory(
    config: dict[str, Any], run_type: str, root: str | Path | None = None,
    timestamp: str | None = None,
) -> Path:
    if run_type not in {"pilot", "full"}:
        raise ValueError("run_type must be pilot or full")
    base = Path(root) if root is not None else PROJECT_ROOT / "results" / "experiments"
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = base / str(config["experiment_name"]) / f"seed_{int(config['seed'])}" / stamp
    output.mkdir(parents=True, exist_ok=False)
    return output


def load_development_manifests(data_config: dict[str, Any], project_root: Path):
    """Load only train and validation manifests; test configuration is intentionally ignored."""
    dataset_root = data_config.get("dataset_root")
    train = validate_manifest(
        load_manifest(data_config["train_manifest"], project_root), "train", project_root, dataset_root
    )
    validation = validate_manifest(
        load_manifest(data_config["val_manifest"], project_root), "val", project_root, dataset_root
    )
    return train, validation


def _load_data_config(config: dict[str, Any]) -> dict[str, Any]:
    relative = Path(str(config["data_config"]))
    if relative.as_posix() != "configs/data_v3_clean.yaml":
        raise ValueError("Stage 4A requires configs/data_v3_clean.yaml")
    with (PROJECT_ROOT / relative).open(encoding="utf-8") as handle:
        data_config = yaml.safe_load(handle)
    if not isinstance(data_config, dict):
        raise ValueError("Data config must be a YAML mapping")
    return data_config


def _compile_model(model: tf.keras.Model, config: dict[str, Any]) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(config["learning_rate"])),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )


def run_training(
    config_path: str, max_epochs: int | None, seed: int | None, run_type: str
) -> Path:
    config = apply_cli_overrides(load_experiment_config(config_path), max_epochs, seed)
    data_config = _load_data_config(config)
    output_dir = create_run_directory(config, run_type)
    config["run_type"] = run_type
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    environment = configure_reproducibility(int(config["seed"]))
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    train_frame, val_frame = load_development_manifests(data_config, PROJECT_ROOT)
    common = {
        "project_root": PROJECT_ROOT,
        "image_size": tuple(config["image_size"]),
        "batch_size": int(config["batch_size"]),
        "seed": int(config["seed"]),
        "dataset_root": data_config.get("dataset_root"),
        "cache": bool(data_config.get("cache", False)),
        "prefetch": bool(data_config.get("prefetch", True)),
    }
    train_dataset = build_dataset(
        data_config["train_manifest"], training=True,
        augment=bool(data_config.get("augmentation", {}).get("enabled", True)),
        expected_split="train", **common,
    )
    val_dataset = build_dataset(
        data_config["val_manifest"], training=False, augment=False,
        expected_split="val", **common,
    )

    model = build_cnn_baseline(
        input_shape=(*tuple(config["image_size"]), 3),
        dropout_rate=float(config["dropout_rate"]),
    )
    _compile_model(model, config)
    summary_buffer = io.StringIO()
    model.summary(print_fn=lambda line: summary_buffer.write(line + "\n"))
    (output_dir / "model_summary.txt").write_text(summary_buffer.getvalue(), encoding="utf-8")

    print(f"Train images: {len(train_frame)}")
    print(f"Validation images: {len(val_frame)}")
    print(f"GPU devices: {environment['gpu_devices']}")
    print(f"Model parameters: {model.count_params():,}")
    print(f"Output directory: {output_dir}")
    if not environment["gpu_devices"]:
        raise RuntimeError("No TensorFlow GPU detected; refusing to start training")

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=int(config["max_epochs"]),
        callbacks=build_callbacks(output_dir, config),
        class_weight=None,
        verbose=2,
    )
    best_path = output_dir / "best_model.keras"
    if not best_path.is_file():
        raise RuntimeError("ModelCheckpoint did not create best_model.keras")
    best_model = tf.keras.models.load_model(best_path)
    metrics = evaluate_validation(
        best_model, val_dataset, val_frame, output_dir, threshold=float(config["threshold"])
    )
    summary = [
        "# CNN baseline run summary", "", f"- Run type: {run_type}",
        f"- Train images: {len(train_frame)}", f"- Validation images: {len(val_frame)}",
        f"- Best checkpoint: `{best_path.name}`", "- Test manifest read: no",
        "- Threshold optimized: no (fixed at 0.5)", "", "## Validation metrics", "",
    ] + [f"- {key}: {value}" for key, value in metrics.items()]
    (output_dir / "run_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"RUN_DIR={output_dir.resolve()}")
    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-type", choices=("pilot", "full"), required=True)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    run_training(args.config, args.max_epochs, args.seed, args.run_type)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
