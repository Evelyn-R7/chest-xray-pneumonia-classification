from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import tensorflow as tf
import yaml

CLASS_MAPPING = {"NORMAL": 0, "PNEUMONIA": 1}
REQUIRED_COLUMNS = {"filepath", "filename", "label", "patient_id", "new_split", "sha256"}
WINDOWS_ABSOLUTE_PATH = re.compile(r"^([A-Za-z]):[\\/](.*)$")


def _resolve_parallel_calls(value: int | str | None):
    if value is None:
        return tf.data.AUTOTUNE
    if isinstance(value, str) and value.lower() == "autotune":
        return tf.data.AUTOTUNE
    calls = int(value)
    if calls < 1:
        raise ValueError("num_parallel_calls must be >= 1 or 'autotune'")
    return calls


def _windows_path_to_wsl(value: str) -> Path | None:
    """Translate a Windows drive path to its conventional WSL mount path."""
    if os.name == "nt":
        return None
    match = WINDOWS_ABSOLUTE_PATH.match(value)
    if not match:
        return None
    drive, remainder = match.groups()
    parts = [part for part in re.split(r"[\\/]+", remainder) if part]
    return Path("/mnt") / drive.lower() / Path(*parts)


def _resolve_path(value: str, project_root: Path, dataset_root: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    wsl_path = _windows_path_to_wsl(value)
    if wsl_path is not None:
        return wsl_path.resolve()
    candidates = [(project_root / path).resolve()]
    if dataset_root is not None:
        candidates.append((dataset_root / path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_manifest(manifest_path: str | Path, project_root: str | Path) -> pd.DataFrame:
    project_root = Path(project_root).resolve()
    path = _resolve_path(str(manifest_path), project_root)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    dataframe = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing:
        raise ValueError(f"Manifest {path} is missing required columns: {sorted(missing)}")
    dataframe.attrs["manifest_path"] = str(path)
    return dataframe


def validate_manifest(
    dataframe: pd.DataFrame,
    expected_split: str,
    project_root: str | Path | None = None,
    dataset_root: str | Path | None = None,
) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    invalid_labels = sorted(set(dataframe["label"]) - set(CLASS_MAPPING))
    if invalid_labels:
        raise ValueError(f"Invalid labels {invalid_labels}; expected {sorted(CLASS_MAPPING)}")
    invalid_splits = sorted(set(dataframe["new_split"]) - {expected_split})
    if invalid_splits:
        raise ValueError(
            f"Manifest expected new_split={expected_split!r}, found {invalid_splits}"
        )
    if project_root is not None:
        root = Path(project_root).resolve()
        data_root = _resolve_path(str(dataset_root), root) if dataset_root else None
        resolved = [_resolve_path(value, root, data_root) for value in dataframe["filepath"]]
        missing_files = [str(path) for path in resolved if not path.is_file()]
        if missing_files:
            preview = "\n".join(missing_files[:10])
            raise FileNotFoundError(f"Manifest references missing files ({len(missing_files)}):\n{preview}")
        dataframe = dataframe.copy()
        dataframe["resolved_filepath"] = [str(path) for path in resolved]
    return dataframe


def decode_image(path: tf.Tensor, image_size: tuple[int, int] | list[int]) -> tf.Tensor:
    contents = tf.io.read_file(path)
    image = tf.io.decode_image(contents, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, image_size, antialias=True)
    image = tf.cast(image, tf.float32)
    image = tf.ensure_shape(image, [image_size[0], image_size[1], 3])
    return image


def build_augmentation(seed: int = 42) -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(factor=0.03, seed=seed),
            tf.keras.layers.RandomTranslation(0.05, 0.05, seed=seed + 1),
            tf.keras.layers.RandomZoom(0.10, 0.10, seed=seed + 2),
            tf.keras.layers.RandomContrast(0.10, seed=seed + 3),
        ],
        name="training_augmentation",
    )


def build_dataset(
    manifest_path: str | Path,
    project_root: str | Path,
    image_size: tuple[int, int] | list[int] = (224, 224),
    batch_size: int = 32,
    training: bool = False,
    seed: int = 42,
    augment: bool = False,
    expected_split: str | None = None,
    dataset_root: str | Path | None = None,
    cache: bool = False,
    prefetch: bool = True,
    num_parallel_calls: int | str | None = None,
    augmentation_num_parallel_calls: int | str | None = None,
) -> tf.data.Dataset:
    project_root = Path(project_root).resolve()
    dataframe = load_manifest(manifest_path, project_root)
    split = expected_split or ("train" if training else str(dataframe["new_split"].iloc[0]))
    dataframe = validate_manifest(dataframe, split, project_root, dataset_root)
    paths = dataframe["resolved_filepath"].to_numpy(dtype=str)
    labels = dataframe["label"].map(CLASS_MAPPING).to_numpy(dtype="float32")
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    options = tf.data.Options()
    options.deterministic = True
    dataset = dataset.with_options(options)
    if training:
        dataset = dataset.shuffle(len(dataframe), seed=seed, reshuffle_each_iteration=True)
    read_parallel_calls = _resolve_parallel_calls(num_parallel_calls)
    dataset = dataset.map(
        lambda path, label: (decode_image(path, image_size), tf.cast(label, tf.float32)),
        num_parallel_calls=read_parallel_calls,
        deterministic=True,
    )
    if cache:
        dataset = dataset.cache()
    dataset = dataset.batch(batch_size, drop_remainder=False)
    if training and augment:
        augmentation = build_augmentation(seed)
        augment_parallel_calls = _resolve_parallel_calls(
            augmentation_num_parallel_calls
            if augmentation_num_parallel_calls is not None
            else num_parallel_calls
        )
        dataset = dataset.map(
            lambda images, labels: (augmentation(images, training=True), labels),
            num_parallel_calls=augment_parallel_calls,
            deterministic=True,
        )
    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def build_all_datasets(config_path: str | Path):
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    with config_path.open(encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    image_size = tuple(config.get("image_size", [224, 224]))
    batch_size = int(config.get("batch_size", 32))
    seed = int(config.get("random_seed", 42))
    augmentation_enabled = bool(config.get("augmentation", {}).get("enabled", True))
    common = dict(
        project_root=project_root, image_size=image_size, batch_size=batch_size, seed=seed,
        dataset_root=config.get("dataset_root"), cache=bool(config.get("cache", False)),
        prefetch=bool(config.get("prefetch", True)),
        num_parallel_calls=config.get("num_parallel_calls"),
        augmentation_num_parallel_calls=config.get("augmentation_num_parallel_calls"),
    )
    train_df = validate_manifest(
        load_manifest(config["train_manifest"], project_root), "train", project_root,
        config.get("dataset_root"),
    )
    val_df = validate_manifest(
        load_manifest(config["val_manifest"], project_root), "val", project_root,
        config.get("dataset_root"),
    )
    test_df = validate_manifest(
        load_manifest(config["test_manifest"], project_root), "test", project_root,
        config.get("dataset_root"),
    )
    train = build_dataset(config["train_manifest"], training=True, augment=augmentation_enabled,
                          expected_split="train", **common)
    val = build_dataset(config["val_manifest"], training=False, augment=False,
                        expected_split="val", **common)
    test = build_dataset(config["test_manifest"], training=False, augment=False,
                         expected_split="test", **common)
    metadata = {
        "train_images": len(train_df), "val_images": len(val_df), "test_images": len(test_df),
        "class_mapping": CLASS_MAPPING.copy(), "image_size": list(image_size),
        "batch_size": batch_size, "positive_class": config.get("positive_class", "PNEUMONIA"),
        "split_protocol": config.get("split_protocol"),
    }
    return train, val, test, metadata
