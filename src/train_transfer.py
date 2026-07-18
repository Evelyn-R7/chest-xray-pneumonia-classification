"""Train VGG16 or EfficientNetB0 in two stages using train/validation only."""

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
    from .models.transfer_models import (
        build_efficientnetb0_transfer, build_vgg16_transfer, freeze_backbone, get_backbone,
        trainable_parameter_count, unfreeze_efficientnet_last_non_bn, unfreeze_vgg16_block5,
    )
    from .training.reproducibility import configure_reproducibility
    from .training.two_stage_training import resolve_stage_epochs, run_two_stage_training
else:
    from data_pipeline import build_dataset, load_manifest, validate_manifest
    from evaluate_validation import evaluate_validation
    from models.transfer_models import (
        build_efficientnetb0_transfer, build_vgg16_transfer, freeze_backbone, get_backbone,
        trainable_parameter_count, unfreeze_efficientnet_last_non_bn, unfreeze_vgg16_block5,
    )
    from training.reproducibility import configure_reproducibility
    from training.two_stage_training import resolve_stage_epochs, run_two_stage_training

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CLASS_COUNTS = {0: 1079, 1: 2742}
TRAIN_TOTAL = 3821


def load_transfer_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Transfer config must be a YAML mapping")
    required = {
        "experiment_name", "model_name", "data_config", "seed", "image_size", "batch_size",
        "dropout_rate", "head_learning_rate", "fine_tune_learning_rate", "head_max_epochs",
        "fine_tune_max_epochs", "head_early_stopping_patience",
        "fine_tune_early_stopping_patience", "reduce_lr_patience", "reduce_lr_factor",
        "minimum_learning_rate", "threshold", "class_weight", "mixed_precision", "weights", "loss",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Transfer config missing keys: {sorted(missing)}")
    if config["model_name"] not in {"vgg16", "efficientnetb0"}:
        raise ValueError("model_name must be vgg16 or efficientnetb0")
    if config["data_config"] != "configs/data_v3_clean.yaml":
        raise ValueError("Stage 5A requires configs/data_v3_clean.yaml")
    if int(config["batch_size"]) != 16:
        raise ValueError("Transfer experiments fix batch_size=16")
    if bool(config["mixed_precision"]):
        raise ValueError("Mixed precision is disabled")
    if float(config["threshold"]) != 0.5:
        raise ValueError("Threshold must remain 0.5")
    validate_imbalance_strategy(config)
    return config


def compute_balanced_class_weight(class_counts: dict[int, int] = TRAIN_CLASS_COUNTS) -> dict[int, float]:
    total = sum(class_counts.values())
    classes = len(class_counts)
    return {label: total / (classes * count) for label, count in class_counts.items()}


def compute_focal_alpha(normal_count: int = TRAIN_CLASS_COUNTS[0], total: int = TRAIN_TOTAL) -> float:
    return normal_count / total


def validate_imbalance_strategy(config: dict[str, Any]) -> None:
    loss = config.get("loss", "binary_crossentropy")
    class_weight = config.get("class_weight")
    focal = config.get("focal_loss")
    if loss not in {"binary_crossentropy", "binary_focal_crossentropy"}:
        raise ValueError(f"Unsupported loss: {loss}")
    if loss == "binary_focal_crossentropy":
        if not isinstance(focal, dict):
            raise ValueError("binary_focal_crossentropy requires focal_loss settings")
        if class_weight is not None and bool(focal.get("apply_class_balancing", False)):
            raise ValueError("Do not combine focal apply_class_balancing with Keras class_weight")
        if bool(focal.get("from_logits", False)):
            raise ValueError("Focal loss must use from_logits=false")
    elif focal is not None:
        raise ValueError("focal_loss settings require loss=binary_focal_crossentropy")


def resolve_imbalance_strategy(config: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(config)
    if resolved.get("class_weight") is not None:
        weights = compute_balanced_class_weight()
        resolved["class_weight"] = {int(label): float(value) for label, value in weights.items()}
    if resolved.get("loss") == "binary_focal_crossentropy":
        focal = dict(resolved.get("focal_loss") or {})
        focal["alpha"] = compute_focal_alpha()
        focal["gamma"] = float(focal.get("gamma", 2.0))
        focal["apply_class_balancing"] = bool(focal.get("apply_class_balancing", True))
        focal["from_logits"] = bool(focal.get("from_logits", False))
        focal["label_smoothing"] = float(focal.get("label_smoothing", 0.0))
        resolved["focal_loss"] = focal
        resolved["class_weight"] = None
    validate_imbalance_strategy(resolved)
    return resolved


def apply_seed_override(config: dict[str, Any], seed: int | None = None) -> dict[str, Any]:
    """Return a copy of the transfer config with an optional CLI seed override."""
    resolved = dict(config)
    if seed is not None:
        resolved["seed"] = int(seed)
    if int(resolved["seed"]) not in {42, 2025, 2026}:
        raise ValueError("Transfer multiseed experiments allow only seeds 42, 2025, and 2026")
    return resolved


def create_transfer_run_directory(config: dict[str, Any], run_type: str, root=None, timestamp=None) -> Path:
    if run_type not in {"pilot", "full"}:
        raise ValueError("run_type must be pilot or full")
    base = Path(root) if root is not None else PROJECT_ROOT / "results" / "experiments"
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = base / config["experiment_name"] / f"seed_{int(config.get('seed', 42))}" / f"{run_type}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def load_development_manifests(data_config: dict[str, Any]):
    root = PROJECT_ROOT
    dataset_root = data_config.get("dataset_root")
    train = validate_manifest(load_manifest(data_config["train_manifest"], root), "train", root, dataset_root)
    val = validate_manifest(load_manifest(data_config["val_manifest"], root), "val", root, dataset_root)
    return train, val


def _build_model_and_unfreezer(config):
    common = ((*tuple(config["image_size"]), 3), float(config["dropout_rate"]), config["weights"])
    if config["model_name"] == "vgg16":
        return build_vgg16_transfer(*common), unfreeze_vgg16_block5
    count = int(config["fine_tune_last_non_bn_layers"])
    return build_efficientnetb0_transfer(*common), lambda model: unfreeze_efficientnet_last_non_bn(model, count)


def run_transfer(config_path: str, run_type: str, seed: int | None = None) -> Path:
    config = resolve_imbalance_strategy(apply_seed_override(load_transfer_config(config_path), seed))
    phase1_epochs, phase2_epochs = resolve_stage_epochs(config, run_type)
    resolved = dict(config, run_type=run_type, resolved_head_epochs=phase1_epochs,
                    resolved_fine_tune_epochs=phase2_epochs)
    output = create_transfer_run_directory(config, run_type)
    print(f"ALLOCATED_RUN_DIR={output.resolve()}", flush=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    environment = configure_reproducibility(int(config["seed"]))
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    data_config = yaml.safe_load((PROJECT_ROOT / config["data_config"]).read_text(encoding="utf-8"))
    train_frame, val_frame = load_development_manifests(data_config)
    labels = train_frame["label"].map({"NORMAL": 0, "PNEUMONIA": 1}).value_counts().to_dict()
    if labels != TRAIN_CLASS_COUNTS:
        raise ValueError(f"Unexpected train class counts: {labels}")
    common = dict(
        project_root=PROJECT_ROOT, image_size=tuple(config["image_size"]), batch_size=int(config["batch_size"]),
        seed=int(config["seed"]), dataset_root=data_config.get("dataset_root"),
        cache=bool(data_config.get("cache", False)),
        prefetch=bool(config.get("data_prefetch", data_config.get("prefetch", True))),
        num_parallel_calls=config.get("data_num_parallel_calls", data_config.get("num_parallel_calls")),
        augmentation_num_parallel_calls=config.get(
            "augmentation_num_parallel_calls",
            data_config.get("augmentation_num_parallel_calls"),
        ),
    )
    train_ds = build_dataset(data_config["train_manifest"], training=True,
        augment=bool(data_config.get("augmentation", {}).get("enabled", True)), expected_split="train", **common)
    val_ds = build_dataset(data_config["val_manifest"], training=False, augment=False, expected_split="val", **common)

    model, unfreeze_fn = _build_model_and_unfreezer(config)
    freeze_backbone(model)
    phase1_trainable = trainable_parameter_count(model)
    summary = io.StringIO(); model.summary(print_fn=lambda line: summary.write(line + "\n"))
    (output / "model_summary.txt").write_text(summary.getvalue(), encoding="utf-8")
    (output / "trainable_layers_phase1.txt").write_text(
        "\n".join(layer.name for layer in model.layers if layer.trainable) + "\n", encoding="utf-8")
    # Preview phase2 state, record exact names/count, then restore phase1 before training.
    unfrozen_preview = unfreeze_fn(model)
    phase2_trainable = trainable_parameter_count(model)
    (output / "trainable_layers_phase2.txt").write_text("\n".join(unfrozen_preview) + "\n", encoding="utf-8")
    freeze_backbone(model)

    print(f"Train images: {len(train_frame)}; Validation images: {len(val_frame)}")
    print(f"GPU devices: {environment['gpu_devices']}")
    print(f"Total parameters: {model.count_params():,}")
    print(f"Phase1 trainable parameters: {phase1_trainable:,}")
    print(f"Phase2 trainable parameters: {phase2_trainable:,}")
    print(f"Unfrozen layers: {unfrozen_preview}")
    print(f"Output directory: {output}")
    if not environment["gpu_devices"]:
        raise RuntimeError("No TensorFlow GPU detected; refusing to train")

    result = run_two_stage_training(model, train_ds, val_ds, output, config, run_type,
                                    freeze_backbone, unfreeze_fn)
    metrics = evaluate_validation(result["model"], val_ds, val_frame, output,
                                  threshold=0.5, history_filename="combined_history.csv")
    lines = [
        f"# {config['experiment_name']} run summary", "", f"- Run type: {run_type}",
        f"- Best phase: {result['best_phase']}",
        f"- Phase1 minimum val_loss: {result['phase1_min_val_loss']}",
        f"- Phase2 minimum val_loss: {result['phase2_min_val_loss']}",
        f"- Train/validation images: {len(train_frame)}/{len(val_frame)}",
        "- Test manifest loaded: no", "- Threshold: 0.5 (not optimized)", "", "## Validation metrics", "",
    ] + [f"- {key}: {value}" for key, value in metrics.items()]
    (output / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"RUN_DIR={output.resolve()}")
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-type", choices=("pilot", "full"), required=True)
    parser.add_argument("--seed", type=int, help="Override YAML seed for registered multiseed runs")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args(); run_transfer(args.config, args.run_type, args.seed); return 0


if __name__ == "__main__":
    raise SystemExit(main())
