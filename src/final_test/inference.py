"""Final test manifest validation, model inference, and ensemble helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_pipeline import CLASS_MAPPING, _windows_path_to_wsl
from src.final_test.safety import sha256_file

EXPECTED_TEST_IMAGES = 624
EXPECTED_TEST_PATIENTS = 427
EXPECTED_TEST_CLASS_COUNTS = {"NORMAL": 234, "PNEUMONIA": 390}
REQUIRED_TEST_COLUMNS = {"filepath", "filename", "patient_id", "label", "new_split", "sha256"}


def validate_probability_array(probabilities: np.ndarray, name: str = "probabilities") -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{name} must contain only finite values in [0, 1]")
    return values


def resolve_manifest_image_path(value: str, project_root: str | Path) -> Path:
    """Resolve manifest image paths, including Windows absolute paths under WSL."""

    wsl_path = _windows_path_to_wsl(value)
    if wsl_path is not None:
        return wsl_path.resolve()
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (Path(project_root) / path).resolve()


def verify_image_can_open(image_path: str | Path) -> None:
    """Verify image readability without modifying the file."""

    from PIL import Image

    with Image.open(image_path) as image:
        image.verify()


def load_and_validate_test_manifest(
    manifest_path: str | Path,
    project_root: str | Path,
    frozen_sha256: str,
    train_manifest: str | Path,
    val_manifest: str | Path,
) -> pd.DataFrame:
    path = Path(manifest_path)
    if sha256_file(path) != frozen_sha256:
        raise ValueError("test.csv SHA-256 does not match frozen protocol")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    validate_test_manifest_frame(frame)
    root = Path(project_root)
    resolved = []
    for value in frame["filepath"]:
        image_path = resolve_manifest_image_path(value, root)
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing test image: {image_path}")
        verify_image_can_open(image_path)
        if sha256_file(image_path) != frame.loc[len(resolved), "sha256"]:
            raise ValueError(f"Image SHA-256 mismatch: {image_path}")
        resolved.append(str(image_path))
    frame = frame.copy()
    frame["resolved_filepath"] = resolved
    train = pd.read_csv(train_manifest, dtype=str, keep_default_na=False)
    val = pd.read_csv(val_manifest, dtype=str, keep_default_na=False)
    train_val_patients = set(train["patient_id"]).union(set(val["patient_id"]))
    overlap = sorted(set(frame["patient_id"]).intersection(train_val_patients))
    if overlap:
        raise ValueError(f"test patient_id overlaps train/val: {overlap[:5]}")
    return frame


def validate_test_manifest_frame(frame: pd.DataFrame) -> None:
    missing = REQUIRED_TEST_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"test manifest missing columns: {sorted(missing)}")
    if len(frame) != EXPECTED_TEST_IMAGES:
        raise ValueError(f"Expected {EXPECTED_TEST_IMAGES} test rows, found {len(frame)}")
    if frame["patient_id"].nunique() != EXPECTED_TEST_PATIENTS:
        raise ValueError(f"Expected {EXPECTED_TEST_PATIENTS} test patients")
    if set(frame["new_split"]) != {"test"}:
        raise ValueError("test manifest must have new_split=test for every row")
    counts = frame["label"].value_counts().to_dict()
    if counts != EXPECTED_TEST_CLASS_COUNTS:
        raise ValueError(f"Unexpected test class counts: {counts}")
    invalid = sorted(set(frame["label"]) - set(CLASS_MAPPING))
    if invalid:
        raise ValueError(f"Invalid labels: {invalid}")
    true_label = frame["label"].map(CLASS_MAPPING).astype(int)
    per_patient = pd.DataFrame({"patient_id": frame["patient_id"], "true_label": true_label}).groupby("patient_id")["true_label"].nunique()
    if (per_patient > 1).any():
        raise ValueError("A patient_id appears with multiple true_label values")


def assert_prediction_alignment(frames: list[pd.DataFrame]) -> None:
    reference = frames[0].reset_index(drop=True)
    for frame in frames[1:]:
        candidate = frame.reset_index(drop=True)
        for column in ("filename", "patient_id", "true_label"):
            if not candidate[column].equals(reference[column]):
                raise ValueError(f"Prediction {column} order/content differs across seeds")


def build_ensemble_prediction_frame(seed_frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    seeds = [42, 2025, 2026]
    frames = [seed_frames[seed].reset_index(drop=True) for seed in seeds]
    assert_prediction_alignment(frames)
    output = frames[0][["filename", "patient_id", "true_label"]].copy()
    probabilities = []
    for seed, frame in zip(seeds, frames):
        values = validate_probability_array(frame[f"probability_seed_{seed}"].to_numpy(float), f"seed {seed} probabilities")
        probabilities.append(values)
        output[f"probability_seed_{seed}"] = values
    matrix = np.column_stack(probabilities)
    output["ensemble_raw_probability"] = matrix.mean(axis=1)
    output["final_probability"] = output["ensemble_raw_probability"]
    output["std_probability"] = matrix.std(axis=1, ddof=1)
    return output


def add_threshold_predictions(frame: pd.DataFrame, benchmark_threshold: float, balanced_threshold: float) -> pd.DataFrame:
    output = frame.copy()
    probs = validate_probability_array(output["final_probability"].to_numpy(float), "final_probability")
    output["predicted_label_threshold_0_5"] = (probs >= benchmark_threshold).astype(int)
    output["predicted_label_balanced_threshold"] = (probs >= balanced_threshold).astype(int)
    return output


def predict_single_model(model_path: str | Path, dataset: Any, expected_count: int = EXPECTED_TEST_IMAGES) -> np.ndarray:
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)
    try:
        probabilities = model.predict(dataset, verbose=1, batch_size=16).reshape(-1)
        if probabilities.size != expected_count:
            raise RuntimeError(f"Expected {expected_count} predictions, found {probabilities.size}")
        return validate_probability_array(probabilities, "model probabilities")
    finally:
        del model
        tf.keras.backend.clear_session()
