from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from src.final_protocol.calibration import (
    apply_calibrator,
    build_oof_calibration_candidates,
    choose_calibration_method,
    expected_calibration_error,
    fit_calibrator,
)
from src.final_protocol.protocol import (
    EXPECTED_SEEDS,
    build_ensemble_predictions,
    load_control_runs,
    sha256_file,
)
from src.final_protocol.thresholds import select_balanced_threshold


def prediction_frame(offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame({
        "filename": ["a.png", "b.png", "c.png", "d.png"],
        "patient_id": ["p1", "p2", "p3", "p4"],
        "true_label": [0, 1, 0, 1],
        "predicted_probability": [0.1 + offset, 0.8 + offset, 0.4 + offset, 0.6 + offset],
        "threshold": [0.5, 0.5, 0.5, 0.5],
    })


def test_three_seed_probability_mean():
    runs = [{"seed": seed, "predictions": prediction_frame(index * 0.01)} for index, seed in enumerate(EXPECTED_SEEDS)]
    ensemble = build_ensemble_predictions(runs)
    assert np.allclose(
        ensemble["ensemble_raw_probability"],
        np.mean([
            prediction_frame(0.0)["predicted_probability"],
            prediction_frame(0.01)["predicted_probability"],
            prediction_frame(0.02)["predicted_probability"],
        ], axis=0),
    )


def test_prediction_order_mismatch_errors():
    runs = [{"seed": 42, "predictions": prediction_frame()}, {"seed": 2025, "predictions": prediction_frame()}]
    runs[1]["predictions"] = runs[1]["predictions"].iloc[[1, 0, 2, 3]].reset_index(drop=True)
    with pytest.raises(ValueError, match="filename"):
        build_ensemble_predictions(runs)


def test_patient_id_mismatch_errors():
    runs = [{"seed": 42, "predictions": prediction_frame()}, {"seed": 2025, "predictions": prediction_frame()}]
    runs[1]["predictions"].loc[0, "patient_id"] = "different"
    with pytest.raises(ValueError, match="patient_id"):
        build_ensemble_predictions(runs)


def test_true_label_mismatch_errors():
    runs = [{"seed": 42, "predictions": prediction_frame()}, {"seed": 2025, "predictions": prediction_frame()}]
    runs[1]["predictions"].loc[0, "true_label"] = 1
    with pytest.raises(ValueError, match="true_label"):
        build_ensemble_predictions(runs)


def test_probability_out_of_range_rejected(tmp_path: Path):
    registry = make_registry(tmp_path)
    bad = tmp_path / "run_42" / "val_predictions.csv"
    frame = pd.read_csv(bad)
    frame.loc[0, "predicted_probability"] = 1.2
    frame.to_csv(bad, index=False)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        load_control_runs(registry)


def test_stratified_group_kfold_no_patient_crossing():
    y = np.array([0, 1] * 20)
    p = np.linspace(0.05, 0.95, 40)
    groups = np.array([f"p{i // 2}" for i in range(40)])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in splitter.split(p, y, groups):
        assert not set(groups[train_idx]).intersection(set(groups[val_idx]))


def test_platt_output_range():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.05, 0.2, 0.7, 0.9, 0.3, 0.8])
    model = fit_calibrator("platt", p, y)
    out = apply_calibrator("platt", model, p)
    assert np.all((out >= 0) & (out <= 1))


def test_isotonic_output_range():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.05, 0.2, 0.7, 0.9, 0.3, 0.8])
    model = fit_calibrator("isotonic", p, y)
    out = apply_calibrator("isotonic", model, p)
    assert np.all((out >= 0) & (out <= 1))


def test_ece_calculation():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert expected_calibration_error(y, p, n_bins=2) == pytest.approx(0.35)


def test_brier_selection_rule():
    summary = {
        "none": {"brier_score": 0.03, "log_loss": 0.2, "ece": 0.1, "roc_auc": 0.9, "pr_auc": 0.9},
        "platt": {"brier_score": 0.02, "log_loss": 0.2, "ece": 0.1, "roc_auc": 0.9, "pr_auc": 0.9},
        "isotonic": {"brier_score": 0.0215, "log_loss": 0.2, "ece": 0.1, "roc_auc": 0.9, "pr_auc": 0.9},
    }
    assert choose_calibration_method(summary) == "platt"


def test_calibration_tie_prefers_simpler_within_001():
    summary = {
        "none": {"brier_score": 0.0209, "log_loss": 0.2, "ece": 0.1, "roc_auc": 0.9, "pr_auc": 0.9},
        "platt": {"brier_score": 0.0200, "log_loss": 0.2, "ece": 0.1, "roc_auc": 0.9, "pr_auc": 0.9},
        "isotonic": {"brier_score": 0.0199, "log_loss": 0.2, "ece": 0.1, "roc_auc": 0.9, "pr_auc": 0.9},
    }
    assert choose_calibration_method(summary) == "none"


def test_balanced_threshold_selection():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    threshold, frame = select_balanced_threshold(y, p)
    assert threshold == pytest.approx(0.6)
    assert frame.iloc[0]["balanced_accuracy"] == pytest.approx(1.0)


def test_threshold_tie_break_prefers_higher_sensitivity_then_closer_to_half():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.2, 0.8, 0.2, 0.8])
    threshold, _ = select_balanced_threshold(y, p)
    assert threshold == pytest.approx(0.2)


def test_final_protocol_contains_three_model_hashes(tmp_path: Path):
    protocol = {"best_model_sha256": {"seed_42": "a", "seed_2025": "b", "seed_2026": "c"}}
    assert len(protocol["best_model_sha256"]) == 3


def test_existing_frozen_protocol_refuses_overwrite(tmp_path: Path):
    from src.build_final_protocol import build_final_protocol

    out = tmp_path / "final_protocol"
    out.mkdir()
    (out / "PROTOCOL_FROZEN").write_text("frozen", encoding="utf-8")
    with pytest.raises(FileExistsError):
        build_final_protocol(tmp_path / "missing.json", out)


def test_code_does_not_load_test_manifest():
    source = Path("src/build_final_protocol.py").read_text(encoding="utf-8")
    assert "read_csv" not in source or "test.csv" not in source


def test_no_training_call_in_final_protocol_code():
    combined = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in ["src/build_final_protocol.py", "src/final_protocol/protocol.py"]
    )
    assert "tensorflow" not in combined.lower()
    assert "load_model" not in combined
    assert ".fit(" not in Path("src/build_final_protocol.py").read_text(encoding="utf-8")


def test_oof_predictions_cover_all_samples_without_patient_leakage():
    y = np.array([0, 1] * 20)
    p = np.linspace(0.05, 0.95, 40)
    groups = np.array([f"p{i // 2}" for i in range(40)])
    predictions, folds, summary = build_oof_calibration_candidates(y, p, groups)
    assert len(predictions) == 40
    assert set(folds["method"]) == {"none", "platt", "isotonic"}
    assert set(summary) == {"none", "platt", "isotonic"}


def make_registry(tmp_path: Path) -> Path:
    runs = {}
    for seed in EXPECTED_SEEDS:
        run_dir = tmp_path / f"run_{seed}"
        run_dir.mkdir()
        config = {
            "experiment_name": "efficientnetb0_transfer_v1",
            "model_name": "efficientnetb0",
            "data_config": "configs/data_v3_clean.yaml",
            "seed": seed,
            "run_type": "full",
            "loss": "binary_crossentropy",
            "class_weight": None,
            "threshold": 0.5,
            "mixed_precision": False,
            "output_dir": str(run_dir),
        }
        (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        frame = pd.DataFrame({
            "filename": [f"{i}.png" for i in range(954)],
            "patient_id": [f"p{i}" for i in range(954)],
            "true_label": [i % 2 for i in range(954)],
            "predicted_probability": np.linspace(0.01, 0.99, 954),
            "threshold": 0.5,
        })
        frame.to_csv(run_dir / "val_predictions.csv", index=False)
        (run_dir / "best_model.keras").write_bytes(f"model-{seed}".encode())
        runs[f"seed_{seed}"] = {"run_dir": str(run_dir)}
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"runs": runs}), encoding="utf-8")
    return registry
