from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.final_test.bootstrap import (
    cluster_bootstrap,
    patient_level_predictions,
    percentile_interval,
    stratified_patient_bootstrap_indices,
    validate_patient_labels,
)
from src.final_test.inference import (
    add_threshold_predictions,
    build_ensemble_prediction_frame,
    resolve_manifest_image_path,
    validate_probability_array,
)
from src.final_test.safety import (
    BENCHMARK_THRESHOLD,
    CONFIRMATION_STRING,
    EXPECTED_BALANCED_THRESHOLD,
    EXPECTED_MODEL_SHA256,
    EXPECTED_PROTOCOL_SHA256,
    check_one_time_markers,
    create_evaluated_marker,
    create_started_marker,
    sha256_file,
    verify_confirmation,
    verify_frozen_marker,
    verify_model_hashes,
    verify_protocol_hash,
)


def synthetic_predictions() -> pd.DataFrame:
    return pd.DataFrame({
        "filename": ["a.png", "b.png", "c.png", "d.png", "e.png", "f.png"],
        "patient_id": ["p0", "p0", "p1", "p2", "p3", "p3"],
        "true_label": [0, 0, 1, 0, 1, 1],
        "final_probability": [0.1, 0.2, 0.8, 0.4, 0.7, 0.9],
    })


def seed_frame(seed: int, offset: float = 0.0) -> pd.DataFrame:
    base = synthetic_predictions()[["filename", "patient_id", "true_label"]].copy()
    base[f"probability_seed_{seed}"] = np.array([0.1, 0.2, 0.8, 0.4, 0.7, 0.9]) + offset
    return base


def test_protocol_hash_mismatch_rejected(tmp_path: Path):
    protocol = tmp_path / "final_protocol.json"
    protocol.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_protocol_hash(protocol, EXPECTED_PROTOCOL_SHA256)


def test_model_hash_mismatch_rejected(tmp_path: Path):
    model = tmp_path / "model.keras"
    model.write_text("wrong", encoding="utf-8")
    protocol = {
        "best_model_paths": {key: str(model) for key in EXPECTED_MODEL_SHA256},
        "best_model_sha256": dict(EXPECTED_MODEL_SHA256),
    }
    with pytest.raises(ValueError, match="On-disk model hash mismatch"):
        verify_model_hashes(protocol)


def test_frozen_marker_mismatch_rejected(tmp_path: Path):
    marker = tmp_path / "PROTOCOL_FROZEN"
    marker.write_text("final_protocol_json_sha256: wrong\n", encoding="utf-8")
    with pytest.raises(ValueError, match="PROTOCOL_FROZEN"):
        verify_frozen_marker(marker)


def test_missing_confirmation_string_rejected():
    with pytest.raises(ValueError, match="confirm"):
        verify_confirmation(EXPECTED_PROTOCOL_SHA256, "NOPE")


def test_existing_test_evaluated_rejected(tmp_path: Path):
    (tmp_path / "TEST_EVALUATED").write_text("done", encoding="utf-8")
    with pytest.raises(FileExistsError, match="TEST_EVALUATED"):
        check_one_time_markers(tmp_path)


def test_existing_started_rejected(tmp_path: Path):
    (tmp_path / "TEST_EVALUATION_STARTED").write_text("started", encoding="utf-8")
    with pytest.raises(FileExistsError, match="TEST_EVALUATION_STARTED"):
        check_one_time_markers(tmp_path)


def test_preflight_does_not_read_test_manifest_content():
    source = Path("src/final_test/preflight.py").read_text(encoding="utf-8")
    assert "read_csv" not in source


def test_preflight_does_not_read_test_images():
    source = Path("src/final_test/preflight.py").read_text(encoding="utf-8")
    forbidden = ["decode_image", "read_file", "PIL", "Image.open", "build_dataset"]
    assert not any(token in source for token in forbidden)


def test_three_model_probability_mean_correct():
    frame = build_ensemble_prediction_frame({
        42: seed_frame(42, 0.0),
        2025: seed_frame(2025, -0.05),
        2026: seed_frame(2026, 0.05),
    })
    expected = np.array([0.1, 0.2, 0.8, 0.4, 0.7, 0.9])
    assert np.allclose(frame["ensemble_raw_probability"], expected)


def test_calibration_none_does_not_change_probability():
    frame = build_ensemble_prediction_frame({
        42: seed_frame(42, 0.0),
        2025: seed_frame(2025, 0.0),
        2026: seed_frame(2026, 0.0),
    })
    assert np.allclose(frame["ensemble_raw_probability"], frame["final_probability"])


def test_two_thresholds_use_frozen_values():
    frame = add_threshold_predictions(synthetic_predictions(), BENCHMARK_THRESHOLD, EXPECTED_BALANCED_THRESHOLD)
    assert "predicted_label_threshold_0_5" in frame
    assert "predicted_label_balanced_threshold" in frame
    assert BENCHMARK_THRESHOLD == 0.5
    assert EXPECTED_BALANCED_THRESHOLD == pytest.approx(0.5618644666666667)


def test_probability_out_of_bounds_errors():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_probability_array(np.array([0.1, 1.2]))


def test_windows_manifest_path_resolves_to_wsl_path():
    resolved = resolve_manifest_image_path(
        r"Z:/example/project/data/raw/chest_xray/test/NORMAL/IM-0001-0001.jpeg",
        "/mnt/z/example/project",
    )
    assert resolved == Path("/mnt/z/example/project/data/raw/chest_xray/test/NORMAL/IM-0001-0001.jpeg")


def test_prediction_order_mismatch_errors():
    bad = seed_frame(2025).iloc[[1, 0, 2, 3, 4, 5]].reset_index(drop=True)
    with pytest.raises(ValueError, match="filename"):
        build_ensemble_prediction_frame({42: seed_frame(42), 2025: bad, 2026: seed_frame(2026)})


def test_patient_id_multiple_labels_errors():
    frame = synthetic_predictions()
    frame.loc[1, "true_label"] = 1
    with pytest.raises(ValueError, match="multiple true_label"):
        validate_patient_labels(frame)


def test_cluster_bootstrap_samples_patients():
    frame = synthetic_predictions()
    rng = np.random.default_rng(1)
    indices = stratified_patient_bootstrap_indices(frame, rng)
    assert set(indices).issubset(set(frame.index))


def test_same_patient_images_remain_grouped_when_sampled():
    frame = synthetic_predictions()
    rng = np.random.default_rng(2)
    indices = stratified_patient_bootstrap_indices(frame, rng)
    sampled = frame.loc[indices]
    patient_counts = sampled["patient_id"].value_counts()
    if "p0" in patient_counts:
        assert patient_counts["p0"] % 2 == 0
    if "p3" in patient_counts:
        assert patient_counts["p3"] % 2 == 0


def test_bootstrap_stratification_keeps_two_classes():
    frame = synthetic_predictions()
    rng = np.random.default_rng(3)
    sample = frame.loc[stratified_patient_bootstrap_indices(frame, rng)]
    assert set(sample["true_label"]) == {0, 1}


def test_bootstrap_replicate_count_verification():
    table, summary = cluster_bootstrap(synthetic_predictions(), 0.5, 0.6, replicates=10, seed=4)
    assert table["replicate"].nunique() == 10
    assert summary["successful_replicates"] == 10
    assert summary["failed_replicates"] == 0


def test_percentile_ci_calculation():
    low, high = percentile_interval(np.arange(100, dtype=float), confidence=0.95)
    assert low == pytest.approx(2.475)
    assert high == pytest.approx(96.525)


def test_patient_level_mean_probability():
    patient = patient_level_predictions(synthetic_predictions())
    p0 = patient.loc[patient["patient_id"] == "p0", "final_probability"].iloc[0]
    assert p0 == pytest.approx(0.15)


def test_no_model_fit_call_in_evaluation_code():
    source = "\n".join(Path(path).read_text(encoding="utf-8") for path in [
        "src/evaluate_final_test_once.py",
        "src/final_test/inference.py",
        "src/final_test/preflight.py",
    ])
    assert ".fit(" not in source


def test_no_train_on_batch_call_in_evaluation_code():
    source = Path("src/evaluate_final_test_once.py").read_text(encoding="utf-8")
    assert "train_on_batch" not in source


def test_evaluation_code_does_not_modify_final_protocol_json():
    source = Path("src/evaluate_final_test_once.py").read_text(encoding="utf-8")
    assert "final_protocol.json" in source
    assert "write_json(protocol_path" not in source
    assert ".write_text(" not in source.split("protocol_path")[0]


def test_evaluation_code_does_not_modify_manifests():
    source = Path("src/evaluate_final_test_once.py").read_text(encoding="utf-8")
    assert "to_csv(root / \"data/splits" not in source
    assert "write_text(root / \"data/splits" not in source
    assert "open(root / \"data/splits" not in source


def test_evaluated_marker_created_only_by_explicit_success_function(tmp_path: Path):
    create_started_marker(tmp_path, {"started": True})
    assert (tmp_path / "TEST_EVALUATION_STARTED").exists()
    assert not (tmp_path / "TEST_EVALUATED").exists()
    create_evaluated_marker(tmp_path, {"done": True})
    assert (tmp_path / "TEST_EVALUATED").exists()


def test_confirmation_accepts_exact_string():
    verify_confirmation(EXPECTED_PROTOCOL_SHA256, CONFIRMATION_STRING)
