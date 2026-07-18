import json
from pathlib import Path

import pandas as pd
import pytest

from src.aggregate_imbalance_strategies import (
    assert_strategy_configs_compatible,
    build_cross_strategy_consistency,
    choose_strategy,
    normalize_strategy_config,
)
from src.train_transfer import (
    apply_seed_override,
    compute_balanced_class_weight,
    compute_focal_alpha,
    create_transfer_run_directory,
    load_transfer_config,
    resolve_imbalance_strategy,
    validate_imbalance_strategy,
)
from src.training.two_stage_training import build_loss, resolve_class_weight


def test_balanced_class_weight_formula():
    weights = compute_balanced_class_weight({0: 1079, 1: 2742})
    assert weights[0] == pytest.approx(1.7706209453)
    assert weights[1] == pytest.approx(0.6967541940)


def test_class_weight_config_resolves_from_train_counts():
    config = load_transfer_config("configs/experiments/efficientnetb0_class_weight_v1.yaml")
    resolved = resolve_imbalance_strategy(config)
    assert resolved["class_weight"][0] == pytest.approx(1.7706209453)
    assert resolved["class_weight"][1] == pytest.approx(0.6967541940)


def test_class_weight_parser_returns_int_keys():
    config = resolve_imbalance_strategy(load_transfer_config("configs/experiments/efficientnetb0_class_weight_v1.yaml"))
    assert resolve_class_weight(config) == {0: pytest.approx(1.7706209453), 1: pytest.approx(0.6967541940)}


def test_focal_alpha_formula():
    assert compute_focal_alpha(1079, 3821) == pytest.approx(0.2823868097)


def test_focal_loss_parameters():
    config = resolve_imbalance_strategy(load_transfer_config("configs/experiments/efficientnetb0_focal_v1.yaml"))
    loss = build_loss(config)
    assert loss.apply_class_balancing is True
    assert loss.alpha == pytest.approx(0.2823868097)
    assert loss.gamma == pytest.approx(2.0)
    assert loss.from_logits is False


def test_focal_and_class_weight_conflict_rejected():
    config = load_transfer_config("configs/experiments/efficientnetb0_focal_v1.yaml")
    config["class_weight"] = {0: 1.0, 1: 1.0}
    with pytest.raises(ValueError, match="Do not combine"):
        validate_imbalance_strategy(config)


def test_phase_strategy_config_is_single_source():
    source = (Path(__file__).resolve().parents[1] / "src/training/two_stage_training.py").read_text(encoding="utf-8")
    assert source.count("compile_binary_model(model, float(config[") == 2
    assert "class_weight=class_weight" in source


def test_validation_dataset_not_weighted():
    source = (Path(__file__).resolve().parents[1] / "src/training/two_stage_training.py").read_text(encoding="utf-8")
    assert "validation_data=validation_dataset" in source
    assert "validation_class_weight" not in source


def test_imbalance_cli_seed_override():
    config = load_transfer_config("configs/experiments/efficientnetb0_focal_v1.yaml")
    assert apply_seed_override(config, 2026)["seed"] == 2026


def test_imbalance_output_directory_not_overwritten(tmp_path):
    config = apply_seed_override(load_transfer_config("configs/experiments/efficientnetb0_class_weight_v1.yaml"), 2025)
    create_transfer_run_directory(config, "full", tmp_path, "same")
    with pytest.raises(FileExistsError):
        create_transfer_run_directory(config, "full", tmp_path, "same")


def test_three_strategy_config_consistency_allows_strategy_fields():
    base = resolve_imbalance_strategy(load_transfer_config("configs/experiments/efficientnetb0_transfer_v1.yaml"))
    weighted = resolve_imbalance_strategy(load_transfer_config("configs/experiments/efficientnetb0_class_weight_v1.yaml"))
    focal = resolve_imbalance_strategy(load_transfer_config("configs/experiments/efficientnetb0_focal_v1.yaml"))
    runs = {
        "control": [{"config": base}],
        "class_weight": [{"config": weighted}],
        "focal": [{"config": focal}],
    }
    assert_strategy_configs_compatible(runs)
    focal["batch_size"] = 8
    with pytest.raises(ValueError, match="differs"):
        assert_strategy_configs_compatible(runs)


def test_normalize_strategy_config_masks_allowed_fields():
    control = {"experiment_name": "a", "loss": "binary_crossentropy", "seed": 42, "batch_size": 16}
    focal = {"experiment_name": "b", "loss": "binary_focal_crossentropy", "seed": 2025, "batch_size": 16}
    assert normalize_strategy_config(control) == normalize_strategy_config(focal)


def test_strategy_selection_rule_prefers_balanced_accuracy():
    frame = pd.DataFrame({
        "strategy": ["control", "class_weight", "focal"],
        "balanced_accuracy_mean": [0.90, 0.91, 0.905],
        "brier_score_mean": [0.02, 0.05, 0.01],
        "balanced_accuracy_sample_std": [0.01, 0.01, 0.01],
        "label_disagreement": [10, 10, 10],
        "all_seeds_incorrect": [5, 5, 5],
        "sensitivity_mean": [0.90, 0.90, 0.90],
        "specificity_mean": [0.90, 0.90, 0.90],
    })
    assert choose_strategy(frame)["selected"] == "class_weight"


def test_strategy_selection_tie_uses_brier_score():
    frame = pd.DataFrame({
        "strategy": ["control", "class_weight", "focal"],
        "balanced_accuracy_mean": [0.900, 0.901, 0.902],
        "brier_score_mean": [0.03, 0.02, 0.04],
        "balanced_accuracy_sample_std": [0.01, 0.01, 0.01],
        "label_disagreement": [10, 10, 10],
        "all_seeds_incorrect": [5, 5, 5],
        "sensitivity_mean": [0.90, 0.90, 0.90],
        "specificity_mean": [0.90, 0.90, 0.90],
    })
    assert choose_strategy(frame)["selected"] == "class_weight"


def test_tradeoff_warning_rule():
    frame = pd.DataFrame({
        "strategy": ["control", "class_weight", "focal"],
        "balanced_accuracy_mean": [0.90, 0.91, 0.89],
        "brier_score_mean": [0.03, 0.02, 0.04],
        "balanced_accuracy_sample_std": [0.01, 0.01, 0.01],
        "label_disagreement": [10, 10, 10],
        "all_seeds_incorrect": [5, 5, 5],
        "sensitivity_mean": [0.95, 0.93, 0.95],
        "specificity_mean": [0.90, 0.90, 0.90],
    })
    result = choose_strategy(frame)
    assert result["selected"] == "class_weight"
    assert result["conclusion"] == "trade_off"


def test_cross_strategy_consistency_aligns_by_sample_key():
    base = pd.DataFrame({
        "filename": ["a.jpeg", "b.jpeg"],
        "patient_id": ["a", "b"],
        "true_label": [0, 1],
        "probability_seed_42": [0.1, 0.9],
        "probability_seed_2025": [0.2, 0.8],
        "probability_seed_2026": [0.3, 0.7],
        "predicted_label_seed_42": [0, 1],
        "predicted_label_seed_2025": [0, 1],
        "predicted_label_seed_2026": [0, 1],
    })
    shuffled = base.iloc[[1, 0]].reset_index(drop=True)
    result = build_cross_strategy_consistency({
        "control": base,
        "class_weight": shuffled,
        "focal": base,
    })
    assert result.loc[0, "filename"] == "a.jpeg"
    assert result.loc[0, "class_weight_mean_probability"] == pytest.approx(0.2)


def test_nonfinite_strategy_metric_rejected():
    with pytest.raises(ValueError):
        json.dumps({"accuracy": float("nan")}, allow_nan=False)


def test_imbalance_sources_do_not_read_test_manifest():
    root = Path(__file__).resolve().parents[1]
    for relative in ["src/train_transfer.py", "src/aggregate_imbalance_strategies.py"]:
        source = (root / relative).read_text(encoding="utf-8")
        assert "test_manifest" not in source
