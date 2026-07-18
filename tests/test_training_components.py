from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from src.models.cnn_baseline import build_cnn_baseline
from src.train_cnn import (
    apply_cli_overrides,
    create_run_directory,
    load_development_manifests,
    load_experiment_config,
)
from src.training.metrics import compute_validation_metrics

ROOT = Path(__file__).resolve().parents[1]


def test_model_input_shape():
    assert build_cnn_baseline().input_shape == (None, 224, 224, 3)


def test_model_output_shape():
    assert build_cnn_baseline(input_shape=(32, 32, 3)).output_shape == (None, 1)


def test_model_output_probability_range():
    model = build_cnn_baseline(input_shape=(16, 16, 3))
    output = model(tf.zeros((2, 16, 16, 3)), training=False).numpy()
    assert np.all((output >= 0.0) & (output <= 1.0))


def test_model_contains_internal_rescaling():
    model = build_cnn_baseline(input_shape=(16, 16, 3))
    assert any(isinstance(layer, tf.keras.layers.Rescaling) for layer in model.layers)


def test_model_has_no_flatten():
    model = build_cnn_baseline(input_shape=(16, 16, 3))
    assert not any(isinstance(layer, tf.keras.layers.Flatten) for layer in model.layers)


def known_metrics():
    # threshold 0.5 -> TN=2, FP=1, FN=1, TP=2
    return compute_validation_metrics([0, 0, 0, 1, 1, 1], [0.1, 0.2, 0.8, 0.4, 0.7, 0.9])


def test_metrics_known_confusion_matrix():
    metrics = known_metrics()
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (2, 1, 1, 2)
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["sensitivity"] == pytest.approx(2 / 3)


def test_specificity():
    assert known_metrics()["specificity"] == pytest.approx(2 / 3)


def test_npv():
    assert known_metrics()["npv"] == pytest.approx(2 / 3)


def test_brier_score():
    expected = np.mean((np.array([0.1, 0.2, 0.8, 0.4, 0.7, 0.9]) - [0, 0, 0, 1, 1, 1]) ** 2)
    assert known_metrics()["brier_score"] == pytest.approx(expected)


def test_invalid_labels_raise():
    with pytest.raises(ValueError, match="binary labels"):
        compute_validation_metrics([0, 2], [0.1, 0.9])


def test_config_parsing():
    config = load_experiment_config("configs/experiments/cnn_baseline_v1.yaml")
    assert config["experiment_name"] == "cnn_baseline_v1"
    assert config["class_weight"] is None
    assert config["mixed_precision"] is False
    assert config["threshold"] == 0.5


def test_cli_max_epochs_override():
    config = load_experiment_config("configs/experiments/cnn_baseline_v1.yaml")
    resolved = apply_cli_overrides(config, 3)
    assert resolved["max_epochs"] == 3
    assert config["max_epochs"] == 30


def test_output_directory_does_not_overwrite(tmp_path):
    config = {"experiment_name": "cnn_baseline_v1", "seed": 42}
    create_run_directory(config, "pilot", tmp_path, timestamp="fixed")
    with pytest.raises(FileExistsError):
        create_run_directory(config, "pilot", tmp_path, timestamp="fixed")


def test_training_manifest_loader_never_reads_test(monkeypatch):
    calls = []
    frame = pd.DataFrame(
        {"filepath": ["unused"], "filename": ["x.jpeg"], "label": ["NORMAL"],
         "patient_id": ["p1"], "new_split": ["train"], "sha256": ["0"]}
    )

    def fake_load(path, project_root):
        calls.append(path)
        result = frame.copy()
        result["new_split"] = "val" if path == "val.csv" else "train"
        return result

    monkeypatch.setattr("src.train_cnn.load_manifest", fake_load)
    monkeypatch.setattr("src.train_cnn.validate_manifest", lambda data, *args: data)
    data_config = {
        "train_manifest": "train.csv",
        "val_manifest": "val.csv",
        "test_manifest": "DO_NOT_READ_test.csv",
        "dataset_root": "unused",
    }
    load_development_manifests(data_config, ROOT)
    assert calls == ["train.csv", "val.csv"]
