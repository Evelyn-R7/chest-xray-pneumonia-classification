import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.aggregate_multiseed import (
    CORE_METRICS,
    REQUIRED_ARTIFACTS,
    assert_configs_equivalent,
    assert_prediction_alignment,
    build_prediction_consistency,
    find_unique_seed42_full,
    summarize_values,
    validate_complete_run,
    _write_plots,
)
from src.train_cnn import apply_cli_overrides, create_run_directory, load_experiment_config


def prediction_frame(label_override=None, patient_override=None):
    rows = 954
    labels = np.tile([0, 1], rows // 2 + 1)[:rows]
    patients = np.array([f"p{i}" for i in range(rows)], dtype=object)
    if label_override is not None:
        labels[0] = label_override
    if patient_override is not None:
        patients[0] = patient_override
    probabilities = np.where(labels == 1, 0.8, 0.2)
    return pd.DataFrame({
        "filepath": [f"/mnt/z/project/data/image_{i}.jpeg" for i in range(rows)],
        "filename": [f"image_{i}.jpeg" for i in range(rows)],
        "patient_id": patients,
        "true_label": labels,
        "predicted_probability": probabilities,
        "predicted_label": (probabilities >= 0.5).astype(int),
        "threshold": 0.5,
    })


def make_run(root: Path, seed: int, name: str = "run", complete: bool = True) -> Path:
    run = root / name
    run.mkdir(parents=True)
    config = load_experiment_config("configs/experiments/cnn_baseline_v1.yaml")
    config.update({"seed": seed, "run_type": "full"})
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run / "environment.json").write_text(json.dumps({"seed": seed}), encoding="utf-8")
    prediction_frame().to_csv(run / "val_predictions.csv", index=False)
    metrics = {key: 0.8 for key in CORE_METRICS}
    metrics["threshold"] = 0.5
    (run / "val_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    pd.DataFrame({"loss": [0.5, 0.4], "val_loss": [0.6, 0.3]}).to_csv(run / "history.csv", index=False)
    for artifact in REQUIRED_ARTIFACTS:
        path = run / artifact
        if not path.exists():
            path.write_bytes(b"x")
    if not complete:
        (run / "best_model.keras").unlink()
    return run


def test_cli_seed_overrides_config():
    config = load_experiment_config("configs/experiments/cnn_baseline_v1.yaml")
    assert apply_cli_overrides(config, seed=2025)["seed"] == 2025
    assert config["seed"] == 42


def test_output_path_contains_seed(tmp_path):
    config = {"experiment_name": "cnn_baseline_v1", "seed": 2026}
    output = create_run_directory(config, "full", tmp_path, timestamp="stamp")
    assert output == tmp_path / "cnn_baseline_v1" / "seed_2026" / "stamp"


def test_run_directory_not_overwritten(tmp_path):
    config = {"experiment_name": "cnn_baseline_v1", "seed": 2025}
    create_run_directory(config, "full", tmp_path, timestamp="same")
    with pytest.raises(FileExistsError):
        create_run_directory(config, "full", tmp_path, timestamp="same")


def test_missing_artifact_rejected(tmp_path):
    run = make_run(tmp_path, 2025, complete=False)
    with pytest.raises(ValueError, match="Incomplete run"):
        validate_complete_run(run, expected_seed=2025)


def test_multiple_seed42_full_runs_rejected(tmp_path):
    make_run(tmp_path, 42, "a")
    make_run(tmp_path, 42, "b")
    with pytest.raises(RuntimeError, match="found 2"):
        find_unique_seed42_full(tmp_path)


def test_configs_must_match_except_seed(tmp_path):
    first = validate_complete_run(make_run(tmp_path, 42, "a"), 42)
    second_dir = make_run(tmp_path, 2025, "b")
    second = validate_complete_run(second_dir, 2025)
    assert_configs_equivalent([first, second])
    second["config"]["batch_size"] = 16
    with pytest.raises(ValueError, match="configurations differ"):
        assert_configs_equivalent([first, second])


def test_prediction_row_count_checked(tmp_path):
    run = make_run(tmp_path, 2025)
    prediction_frame().iloc[:-1].to_csv(run / "val_predictions.csv", index=False)
    with pytest.raises(ValueError, match="954"):
        validate_complete_run(run, 2025)


def test_true_label_mismatch_rejected():
    frames = [prediction_frame(), prediction_frame(label_override=1), prediction_frame()]
    with pytest.raises(ValueError, match="true_label"):
        assert_prediction_alignment(frames)


def test_patient_id_mismatch_rejected():
    frames = [prediction_frame(), prediction_frame(patient_override="different"), prediction_frame()]
    with pytest.raises(ValueError, match="patient_id"):
        assert_prediction_alignment(frames)


def test_mean_and_sample_standard_deviation():
    summary = summarize_values([1.0, 2.0, 3.0])
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["sample_std"] == pytest.approx(1.0)


def test_incorrect_count():
    frames = [prediction_frame() for _ in range(3)]
    frames[0].loc[0, "predicted_probability"] = 0.9  # true label is 0
    runs = [{"predictions": frame, "config": {"seed": seed}} for frame, seed in zip(frames, (42, 2025, 2026))]
    consistency = build_prediction_consistency(runs)
    assert consistency.loc[0, "incorrect_count"] == 1
    assert bool(consistency.loc[0, "all_seeds_correct"]) is False


def test_nonfinite_metric_rejected(tmp_path):
    run = make_run(tmp_path, 2025)
    metrics = json.loads((run / "val_metrics.json").read_text(encoding="utf-8"))
    metrics["accuracy"] = float("nan")
    (run / "val_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        validate_complete_run(run, 2025)


def test_aggregation_source_does_not_reference_test_manifest():
    source = (Path(__file__).resolve().parents[1] / "src" / "aggregate_multiseed.py").read_text(encoding="utf-8")
    assert "test_manifest" not in source


def test_both_summary_plots_are_created(tmp_path):
    metrics = pd.DataFrame({
        "seed": [42, 2025, 2026],
        "accuracy": [0.8, 0.9, 0.85],
        "sensitivity": [0.8, 0.9, 0.85],
        "specificity": [0.8, 0.9, 0.85],
        "balanced_accuracy": [0.8, 0.9, 0.85],
        "roc_auc": [0.8, 0.9, 0.85],
        "pr_auc": [0.8, 0.9, 0.85],
    })
    consistency = pd.DataFrame({"std_probability": [0.01, 0.10, 0.20]})
    _write_plots(metrics, consistency, tmp_path)
    assert (tmp_path / "multiseed_metrics.png").stat().st_size > 0
    assert (tmp_path / "probability_variability.png").stat().st_size > 0
