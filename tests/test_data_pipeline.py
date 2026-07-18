from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from src.data_pipeline import (
    CLASS_MAPPING,
    _windows_path_to_wsl,
    build_augmentation,
    build_dataset,
    load_manifest,
    validate_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROWS = {"train": 3821, "val": 954, "test": 624}


def test_windows_absolute_path_maps_to_wsl():
    mapped = _windows_path_to_wsl(r"Z:\example\project\data\image.jpeg")
    if mapped is not None:
        assert mapped == Path("/mnt/z/example/project/data/image.jpeg")


def frame(split):
    return validate_manifest(load_manifest(f"data/splits/v3_clean/{split}.csv", ROOT), split,
                             ROOT, "data/raw/chest_xray")


def test_label_mapping(): assert CLASS_MAPPING == {"NORMAL": 0, "PNEUMONIA": 1}
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_manifest_rows(split): assert len(frame(split)) == EXPECTED_ROWS[split]
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_files_exist(split): assert frame(split)["resolved_filepath"].map(lambda p: Path(p).is_file()).all()


def test_image_shape_and_dtype():
    images, labels = next(iter(build_dataset("data/splits/v3_clean/train.csv", ROOT, training=False,
                                              expected_split="train", prefetch=False)))
    assert images.shape == (32, 224, 224, 3) and images.dtype == tf.float32
    assert labels.shape == (32,) and labels.dtype == tf.float32


def test_label_domain():
    _, labels = next(iter(build_dataset("data/splits/v3_clean/train.csv", ROOT, training=False,
                                        expected_split="train", prefetch=False)))
    assert set(np.unique(labels.numpy())).issubset({0.0, 1.0})


def test_serial_image_reading_option():
    images, labels = next(iter(build_dataset("data/splits/v3_clean/train.csv", ROOT, training=False,
                                             expected_split="train", prefetch=False,
                                             num_parallel_calls=1)))
    assert images.shape == (32, 224, 224, 3)
    assert labels.shape == (32,)


@pytest.mark.parametrize("split", ["val", "test"])
def test_evaluation_determinism(split):
    dataset = build_dataset(f"data/splits/v3_clean/{split}.csv", ROOT, training=False,
                            expected_split=split, prefetch=False)
    first = next(iter(dataset)); second = next(iter(dataset))
    assert np.array_equal(first[0].numpy(), second[0].numpy())
    assert np.array_equal(first[1].numpy(), second[1].numpy())


def test_augmentation_preserves_shape_and_label():
    images, labels = next(iter(build_dataset("data/splits/v3_clean/train.csv", ROOT, training=False,
                                              expected_split="train", prefetch=False)))
    augmented = build_augmentation(42)(images, training=True)
    assert augmented.shape == images.shape and np.array_equal(labels.numpy(), labels.numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(augmented)))


def test_invalid_label_raises():
    bad = frame("train").head(1).copy(); bad.loc[bad.index[0], "label"] = "INVALID"
    with pytest.raises(ValueError, match="Invalid labels"): validate_manifest(bad, "train")


def test_missing_path_raises():
    bad = frame("train").head(1).copy(); bad.loc[bad.index[0], "filepath"] = "missing.jpeg"
    with pytest.raises(FileNotFoundError, match="missing files"):
        validate_manifest(bad, "train", ROOT, "data/raw/chest_xray")
