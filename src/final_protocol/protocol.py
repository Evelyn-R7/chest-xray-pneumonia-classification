"""Final protocol artifact helpers that do not load model or test data."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import log_loss

from src.training.metrics import compute_validation_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SEEDS = (42, 2025, 2026)
EXPECTED_VALIDATION_ROWS = 954
FIXED_THRESHOLD = 0.5
MANIFESTS = (
    "data/splits/v3_clean/train.csv",
    "data/splits/v3_clean/val.csv",
    "data/splits/v3_clean/test.csv",
    "data/splits/v3_clean/excluded_from_development.csv",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def registry_run_dirs(registry_path: str | Path) -> dict[int, Path]:
    registry = read_json(registry_path)
    runs = registry.get("runs")
    if not isinstance(runs, dict):
        raise ValueError("Registry must contain a runs object")
    run_dirs: dict[int, Path] = {}
    for seed in EXPECTED_SEEDS:
        entry = runs.get(f"seed_{seed}")
        if not isinstance(entry, dict) or "run_dir" not in entry:
            raise ValueError(f"Registry missing seed_{seed} run_dir")
        run_dirs[seed] = resolve_project_path(entry["run_dir"]).resolve()
    return run_dirs


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    normalized.setdefault("loss", "binary_crossentropy")
    ignored = {"seed", "run_timestamp", "output_dir"}
    return {key: value for key, value in normalized.items() if key not in ignored}


def validate_control_config(config: dict[str, Any], seed: int, run_dir: Path) -> None:
    if int(config.get("seed")) != seed:
        raise ValueError(f"Seed mismatch for {run_dir}")
    if config.get("experiment_name") != "efficientnetb0_transfer_v1":
        raise ValueError(f"Run is not EfficientNetB0 control: {run_dir}")
    if config.get("model_name") != "efficientnetb0":
        raise ValueError(f"Run is not EfficientNetB0: {run_dir}")
    if config.get("data_config") != "configs/data_v3_clean.yaml":
        raise ValueError(f"Run does not use v3_clean: {run_dir}")
    if config.get("run_type") != "full":
        raise ValueError(f"Run is not a full run: {run_dir}")
    if config.get("loss", "binary_crossentropy") != "binary_crossentropy":
        raise ValueError(f"Run loss is not Binary Crossentropy: {run_dir}")
    if config.get("class_weight") is not None:
        raise ValueError(f"Control run must have class_weight=null: {run_dir}")
    if float(config.get("threshold")) != FIXED_THRESHOLD:
        raise ValueError(f"Control run threshold is not 0.5: {run_dir}")
    if bool(config.get("mixed_precision")):
        raise ValueError(f"Mixed precision is forbidden in final protocol: {run_dir}")


def load_control_runs(registry_path: str | Path) -> list[dict[str, Any]]:
    run_dirs = registry_run_dirs(registry_path)
    runs: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        run_dir = run_dirs[seed]
        config_path = run_dir / "resolved_config.yaml"
        predictions_path = run_dir / "val_predictions.csv"
        model_path = run_dir / "best_model.keras"
        for path in (config_path, predictions_path, model_path):
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing or empty required artifact: {path}")
        config = read_yaml(config_path)
        validate_control_config(config, seed, run_dir)
        predictions = pd.read_csv(predictions_path)
        validate_prediction_frame(predictions, seed, predictions_path)
        runs.append({
            "seed": seed,
            "run_dir": run_dir,
            "config": config,
            "predictions": predictions,
            "config_path": config_path,
            "predictions_path": predictions_path,
            "model_path": model_path,
        })
    reference = normalize_config(runs[0]["config"])
    for run in runs[1:]:
        if normalize_config(run["config"]) != reference:
            raise ValueError("Resolved configs differ by more than seed/timestamp/output path")
    assert_prediction_alignment([run["predictions"] for run in runs])
    return runs


def validate_prediction_frame(frame: pd.DataFrame, seed: int, path: str | Path) -> None:
    required = {"filename", "patient_id", "true_label", "predicted_probability", "threshold"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction file for seed {seed} is missing columns {sorted(missing)}: {path}")
    if len(frame) != EXPECTED_VALIDATION_ROWS:
        raise ValueError(f"Expected 954 validation rows for seed {seed}, found {len(frame)}")
    if not set(frame["true_label"].astype(int).unique()).issubset({0, 1}):
        raise ValueError(f"Invalid labels in prediction file for seed {seed}")
    probabilities = frame["predicted_probability"].to_numpy(dtype=float)
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError(f"Probabilities for seed {seed} must be finite values in [0, 1]")
    if not np.allclose(frame["threshold"].astype(float).to_numpy(), FIXED_THRESHOLD):
        raise ValueError(f"Prediction threshold for seed {seed} is not fixed at 0.5")


def assert_prediction_alignment(frames: list[pd.DataFrame]) -> None:
    reference = frames[0].reset_index(drop=True)
    for frame in frames[1:]:
        candidate = frame.reset_index(drop=True)
        for column in ("filename", "patient_id", "true_label"):
            if not candidate[column].equals(reference[column]):
                raise ValueError(f"Validation {column} order/content differs across seeds")


def build_ensemble_predictions(runs: list[dict[str, Any]]) -> pd.DataFrame:
    frames = [run["predictions"].reset_index(drop=True) for run in runs]
    assert_prediction_alignment(frames)
    output = frames[0][["filename", "patient_id", "true_label"]].copy()
    probabilities = []
    for run, frame in zip(runs, frames):
        seed = int(run["seed"])
        values = frame["predicted_probability"].to_numpy(dtype=float)
        probabilities.append(values)
        output[f"probability_seed_{seed}"] = values
    matrix = np.column_stack(probabilities)
    output["ensemble_raw_probability"] = matrix.mean(axis=1)
    output["ensemble_raw_label_0_5"] = (output["ensemble_raw_probability"].to_numpy(float) >= FIXED_THRESHOLD).astype(int)
    output["std_probability"] = matrix.std(axis=1, ddof=1)
    if "filepath" in output.columns:
        raise ValueError("Final ensemble output must not contain local absolute filepath")
    return output


def metrics_with_log_loss(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    metrics = compute_validation_metrics(y_true, probabilities, threshold=threshold)
    metrics["log_loss"] = float(log_loss(y_true, np.clip(probabilities, 1e-7, 1 - 1e-7), labels=[0, 1]))
    return metrics
