from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from src.posthoc_error_analysis import (
    PRIMARY_THRESHOLD,
    add_error_groups,
    ensure_primary_threshold,
    gradcam_heatmap,
    select_gradcam_samples,
)


def synthetic_predictions() -> pd.DataFrame:
    rows = []
    for i in range(152):
        rows.append(("tn", i, 0, 0.1, 0.11, 0.12))
    for i in range(82):
        rows.append(("fp", i, 0, 0.91 if i < 3 else 0.6, 0.9 if i < 3 else 0.58, 0.92 if i < 3 else 0.57))
    rows.append(("fn", 0, 1, 0.08, 0.09, 0.07))
    for i in range(389):
        rows.append(("tp", i, 1, 0.95, 0.94, 0.96))
    frame = pd.DataFrame(rows, columns=["prefix", "i", "true_label", "probability_seed_42", "probability_seed_2025", "probability_seed_2026"])
    frame["filename"] = frame["prefix"] + frame["i"].astype(str) + ".jpeg"
    frame["patient_id"] = frame["prefix"] + frame["i"].astype(str)
    frame["final_probability"] = frame[["probability_seed_42", "probability_seed_2025", "probability_seed_2026"]].mean(axis=1)
    frame["ensemble_raw_probability"] = frame["final_probability"]
    frame["std_probability"] = frame[["probability_seed_42", "probability_seed_2025", "probability_seed_2026"]].std(axis=1)
    return frame


def test_confusion_group_counts_verify():
    grouped = add_error_groups(synthetic_predictions())
    assert grouped["error_group"].value_counts().to_dict() == {"TP": 389, "TN": 152, "FP": 82, "FN": 1}


def test_primary_threshold_fixed():
    assert PRIMARY_THRESHOLD == pytest.approx(0.5618644666666667)


def test_other_threshold_rejected():
    with pytest.raises(ValueError):
        ensure_primary_threshold(0.5)


def test_high_confidence_and_near_threshold_definitions():
    grouped = add_error_groups(synthetic_predictions())
    assert int(((grouped["error_group"] == "FP") & (grouped["final_probability"] >= 0.90)).sum()) == 3
    assert int(((grouped["error_group"] == "FP") & grouped["near_threshold"]).sum()) == 79


def test_selection_rule_deterministic_and_fn_included():
    grouped = add_error_groups(synthetic_predictions())
    first = select_gradcam_samples(grouped)
    second = select_gradcam_samples(grouped)
    assert first["filename"].tolist() == second["filename"].tolist()
    assert (first["error_group"] == "FN").sum() == 1


def test_duplicate_selection_dedupes_and_records_reasons():
    grouped = add_error_groups(synthetic_predictions())
    selected = select_gradcam_samples(grouped)
    assert selected["filename"].is_unique
    assert selected["selection_reason"].str.contains(";").any()


def test_outputs_do_not_contain_absolute_paths():
    grouped = add_error_groups(synthetic_predictions())
    selected = select_gradcam_samples(grouped)
    assert not any("path" in column.lower() for column in selected.columns)


def test_gradcam_code_does_not_call_fit_or_full_predict():
    source = Path("src/posthoc_error_analysis.py").read_text(encoding="utf-8")
    assert ".fit(" not in source
    assert "model.predict(" not in source


def test_does_not_modify_frozen_protocol_or_test_evaluated():
    source = Path("src/posthoc_error_analysis.py").read_text(encoding="utf-8")
    assert "final_protocol.json" in source
    assert "PROTOCOL_FROZEN" not in source
    assert "TEST_EVALUATED" not in source
    assert "write_text(FINAL_PROTOCOL_PATH" not in source


def test_synthetic_gradcam_shape():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8, 8, 3)),
        tf.keras.layers.Conv2D(2, 3, activation="relu", name="conv"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    image = np.ones((8, 8, 3), dtype=np.float32)
    heatmap = gradcam_heatmap(model, image, model.get_layer("conv"))
    assert heatmap.shape == (6, 6)
    assert np.all(np.isfinite(heatmap))
