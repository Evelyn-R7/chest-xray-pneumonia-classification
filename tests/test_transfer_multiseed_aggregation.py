import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.aggregate_transfer_multiseed import (
    CORE_METRICS,
    REQUIRED_ARTIFACTS,
    assert_configs_equivalent,
    assert_prediction_alignment,
    build_metrics_summary,
    build_prediction_consistency,
    summarize_values,
    validate_complete_transfer_run,
)
from src.train_transfer import apply_seed_override, create_transfer_run_directory, load_transfer_config


def base_config(seed=42):
    config = load_transfer_config("configs/experiments/efficientnetb0_transfer_v1.yaml")
    return apply_seed_override(config, seed)


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
    config = base_config(seed)
    config.update({
        "run_type": "full",
        "resolved_head_epochs": 8,
        "resolved_fine_tune_epochs": 20,
    })
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run / "environment.json").write_text(json.dumps({"seed": seed}), encoding="utf-8")
    prediction_frame().to_csv(run / "val_predictions.csv", index=False)
    metrics = {key: 0.8 for key in CORE_METRICS}
    metrics.update({"threshold": 0.5, "tn": 250, "fp": 20, "fn": 30, "tp": 654})
    (run / "val_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    pd.DataFrame({"epoch": [0, 1], "loss": [0.5, 0.4], "val_loss": [0.6, 0.3]}).to_csv(
        run / "phase1_history.csv", index=False
    )
    pd.DataFrame({"epoch": [2, 3], "loss": [0.35, 0.25], "val_loss": [0.28, 0.24]}).to_csv(
        run / "phase2_history.csv", index=False
    )
    pd.DataFrame({
        "epoch": [0, 1, 2, 3],
        "loss": [0.5, 0.4, 0.35, 0.25],
        "val_loss": [0.6, 0.3, 0.28, 0.24],
    }).to_csv(run / "combined_history.csv", index=False)
    (run / "run_summary.md").write_text("- Best phase: phase2\n", encoding="utf-8")
    (run / "console.log").write_text("239/239 - 10s - loss: 0.1\n", encoding="utf-8")
    for artifact in REQUIRED_ARTIFACTS:
        path = run / artifact
        if not path.exists():
            path.write_bytes(b"x")
    if not complete:
        (run / "phase2_best.keras").unlink()
    return run


def test_transfer_cli_seed_overrides_config():
    config = load_transfer_config("configs/experiments/efficientnetb0_transfer_v1.yaml")
    updated = apply_seed_override(config, 2025)
    assert updated["seed"] == 2025
    assert config["seed"] == 42


def test_transfer_output_path_contains_seed(tmp_path):
    output = create_transfer_run_directory(base_config(2026), "full", tmp_path, timestamp="stamp")
    assert output == tmp_path / "efficientnetb0_transfer_v1" / "seed_2026" / "full_stamp"


def test_transfer_run_directory_not_overwritten(tmp_path):
    config = base_config(2025)
    create_transfer_run_directory(config, "full", tmp_path, timestamp="same")
    with pytest.raises(FileExistsError):
        create_transfer_run_directory(config, "full", tmp_path, timestamp="same")


def test_missing_phase_artifact_rejected(tmp_path):
    run = make_run(tmp_path, 2025, complete=False)
    with pytest.raises(ValueError, match="Incomplete transfer run"):
        validate_complete_transfer_run(run, expected_seed=2025)


def test_transfer_prediction_row_count_checked(tmp_path):
    run = make_run(tmp_path, 2025)
    prediction_frame().iloc[:-1].to_csv(run / "val_predictions.csv", index=False)
    with pytest.raises(ValueError, match="954"):
        validate_complete_transfer_run(run, 2025)


def test_transfer_configs_must_match_except_seed(tmp_path):
    first = validate_complete_transfer_run(make_run(tmp_path, 42, "a"), 42)
    second = validate_complete_transfer_run(make_run(tmp_path, 2025, "b"), 2025)
    assert_configs_equivalent([first, second])
    second["config"]["batch_size"] = 8
    with pytest.raises(ValueError, match="configurations differ"):
        assert_configs_equivalent([first, second])


def test_transfer_true_label_mismatch_rejected():
    frames = [prediction_frame(), prediction_frame(label_override=1), prediction_frame()]
    with pytest.raises(ValueError, match="true_label"):
        assert_prediction_alignment(frames)


def test_transfer_patient_id_mismatch_rejected():
    frames = [prediction_frame(), prediction_frame(patient_override="different"), prediction_frame()]
    with pytest.raises(ValueError, match="patient_id"):
        assert_prediction_alignment(frames)


def test_transfer_best_phase_legality_checked(tmp_path):
    run = make_run(tmp_path, 2025)
    (run / "run_summary.md").write_text("- Best phase: phase3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Best phase"):
        validate_complete_transfer_run(run, 2025)


def test_transfer_mean_and_sample_standard_deviation():
    summary = summarize_values([1.0, 2.0, 3.0])
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["sample_std"] == pytest.approx(1.0)


def test_transfer_incorrect_count_and_label_disagreement():
    frames = [prediction_frame() for _ in range(3)]
    frames[0].loc[0, "predicted_probability"] = 0.9
    frames[1].loc[0, "predicted_probability"] = 0.9
    runs = [{"predictions": frame, "config": {"seed": seed}} for frame, seed in zip(frames, (42, 2025, 2026))]
    consistency = build_prediction_consistency(runs)
    assert consistency.loc[0, "incorrect_count"] == 2
    assert bool(consistency.loc[0, "label_disagreement"]) is True
    assert bool(consistency.loc[0, "all_seeds_correct"]) is False


def test_transfer_nonfinite_metric_rejected(tmp_path):
    run = make_run(tmp_path, 2025)
    metrics = json.loads((run / "val_metrics.json").read_text(encoding="utf-8"))
    metrics["accuracy"] = float("nan")
    (run / "val_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        validate_complete_transfer_run(run, 2025)


def test_transfer_metrics_summary_contains_sample_std(tmp_path):
    rows = []
    for seed, accuracy in zip((42, 2025, 2026), (0.8, 0.9, 1.0)):
        run = validate_complete_transfer_run(make_run(tmp_path, seed, f"seed_{seed}"), seed)
        row = {
            "seed": seed, "best_phase": "phase2", "accuracy": accuracy,
            "precision": 0.8, "sensitivity": 0.8, "specificity": 0.8, "f1": 0.8,
            "balanced_accuracy": 0.8, "roc_auc": 0.8, "pr_auc": 0.8, "npv": 0.8,
            "brier_score": 0.8, "val_loss": 0.2, "best_epoch": 4,
            "phase1_epochs": 2, "phase2_epochs": 2, "total_training_epochs": 4,
            "training_duration": run["training_duration"], "tn": 250, "fp": 20, "fn": 30, "tp": 654,
        }
        rows.append(row)
    summary = build_metrics_summary(pd.DataFrame(rows))
    assert summary["accuracy"]["mean"] == pytest.approx(0.9)
    assert summary["accuracy"]["sample_std"] == pytest.approx(0.1)


def test_transfer_aggregation_source_does_not_reference_test_manifest():
    source = (Path(__file__).resolve().parents[1] / "src" / "aggregate_transfer_multiseed.py").read_text(
        encoding="utf-8"
    )
    assert "test_manifest" not in source
