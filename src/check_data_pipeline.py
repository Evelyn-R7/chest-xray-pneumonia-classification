from __future__ import annotations

import json
import hashlib
import platform
import sys
import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data_pipeline import (
    CLASS_MAPPING, build_all_datasets, build_augmentation, build_dataset,
    decode_image, load_manifest, validate_manifest,
)

EXPECTED = {
    "train": (3821, {"NORMAL": 1079, "PNEUMONIA": 2742}),
    "val": (954, {"NORMAL": 270, "PNEUMONIA": 684}),
    "test": (624, {"NORMAL": 234, "PNEUMONIA": 390}),
}


def assert_batch(images, labels, expected_size=32):
    assert images.shape == (expected_size, 224, 224, 3), images.shape
    assert labels.shape == (expected_size,), labels.shape
    assert images.dtype == tf.float32 and labels.dtype == tf.float32
    assert bool(tf.reduce_all(tf.math.is_finite(images)))
    assert set(np.unique(labels.numpy())).issubset({0.0, 1.0})
    minimum, maximum = float(tf.reduce_min(images)), float(tf.reduce_max(images))
    assert 0.0 <= minimum <= 255.0 and 0.0 <= maximum <= 255.0
    return minimum, maximum


def traverse(dataset):
    count, last = 0, None
    for images, labels in dataset:
        count += int(labels.shape[0])
        last = int(labels.shape[0])
        assert bool(tf.reduce_all(tf.math.is_finite(images)))
    return count, last


def deterministic_twice(dataset):
    def signatures():
        result = []
        for images, labels in dataset:
            result.append((hashlib.sha256(images.numpy().tobytes()).hexdigest(), labels.numpy().tobytes()))
        return result
    first = signatures()
    second = signatures()
    assert len(first) == len(second)
    assert first == second
    return sum(len(labels) // np.dtype("float32").itemsize for _, labels in first)


def plot_batch(images, frame, path, title):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    reverse = {value: key for key, value in CLASS_MAPPING.items()}
    for index, axis in enumerate(axes.flat):
        axis.imshow(np.clip(images[index].numpy(), 0, 255).astype("uint8"))
        row = frame.iloc[index]
        axis.set_title(f"{reverse[CLASS_MAPPING[row['label']]]} | {row['patient_id']}\n{row['filename']}", fontsize=7)
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Check the v3_clean TensorFlow data pipeline and write local QA artifacts.")
    parser.add_argument("--config", default="configs/data_v3_clean.yaml", help="Data config path.")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config = project_root / args.config
    output_dir = project_root / "reports" / "data_pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)
    train, val, test, metadata = build_all_datasets(config)
    frames = {}
    for split in EXPECTED:
        frame = validate_manifest(
            load_manifest(f"data/splits/v3_clean/{split}.csv", project_root), split,
            project_root, "data/raw/chest_xray",
        )
        expected_rows, expected_classes = EXPECTED[split]
        assert len(frame) == expected_rows
        assert frame["label"].value_counts().to_dict() == expected_classes
        frames[split] = frame

    train_images, train_labels = next(iter(train))
    pixel_min, pixel_max = assert_batch(train_images, train_labels)
    counts = {}
    counts["train"], train_last = traverse(train)
    counts["val"] = deterministic_twice(val)
    counts["test"] = deterministic_twice(test)
    assert counts == {name: values[0] for name, values in EXPECTED.items()}
    assert train_last == EXPECTED["train"][0] % 32
    _, val_last = traverse(val)
    _, test_last = traverse(test)
    assert val_last == EXPECTED["val"][0] % 32 and test_last == EXPECTED["test"][0] % 32

    plain_train = build_dataset(
        "data/splits/v3_clean/train.csv", project_root, training=False, augment=False,
        expected_split="train", batch_size=32, prefetch=False,
    )
    original, original_labels = next(iter(plain_train))
    original_again, labels_again = next(iter(plain_train))
    assert np.array_equal(original.numpy(), original_again.numpy())
    assert np.array_equal(original_labels.numpy(), labels_again.numpy())
    augmentation = build_augmentation(42)
    augmented_1 = augmentation(original, training=True)
    augmented_2 = augmentation(original, training=True)
    assert augmented_1.shape == original.shape == augmented_2.shape
    assert not np.array_equal(augmented_1.numpy(), augmented_2.numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(augmented_1)))

    plot_batch(original, frames["train"], output_dir / "train_batch_original.png", "Train original")
    plot_batch(augmented_1, frames["train"], output_dir / "train_batch_augmented.png", "Train augmented")
    val_images, _ = next(iter(val))
    test_images, _ = next(iter(test))
    plot_batch(val_images, frames["val"], output_dir / "val_batch.png", "Validation")
    plot_batch(test_images, frames["test"], output_dir / "test_batch.png", "Test")

    result = {
        "python_version": platform.python_version(), "tensorflow_version": tf.__version__,
        "gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
        "metadata": metadata, "traversed_samples": counts,
        "first_train_batch": {"image_shape": list(train_images.shape), "label_shape": list(train_labels.shape),
                              "image_dtype": train_images.dtype.name, "label_dtype": train_labels.dtype.name,
                              "pixel_min": pixel_min, "pixel_max": pixel_max},
        "last_batch_sizes": {"train": train_last, "val": val_last, "test": test_last},
        "validation_deterministic": True, "test_deterministic": True,
        "augmentation_changes_images": True, "augmentation_preserves_shape_and_labels": True,
        "visualizations": [str(path.relative_to(project_root)).replace("\\", "/") for path in sorted(output_dir.glob("*.png"))],
    }
    (output_dir / "check_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
