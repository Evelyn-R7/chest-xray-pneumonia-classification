"""Compare EfficientNetB0 imbalance strategies from registered validation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__:
    from .aggregate_transfer_multiseed import (
        CORE_METRICS,
        EXPECTED_SEEDS,
        PROJECT_ROOT,
        aggregate_transfer_registry,
        build_metrics_summary,
        build_prediction_consistency,
        load_registered_runs,
        markdown_table,
        run_metrics,
    )
else:
    from aggregate_transfer_multiseed import (
        CORE_METRICS,
        EXPECTED_SEEDS,
        PROJECT_ROOT,
        aggregate_transfer_registry,
        build_metrics_summary,
        build_prediction_consistency,
        load_registered_runs,
        markdown_table,
        run_metrics,
    )

STRATEGIES = ("control", "class_weight", "focal")
IGNORED_CONFIG_FIELDS = {"experiment_name", "loss", "class_weight", "focal_loss", "seed", "run_timestamp", "output_dir"}
SELECTION_TOLERANCE = 0.002


def normalize_strategy_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = {key: value for key, value in config.items() if key not in IGNORED_CONFIG_FIELDS}
    normalized["loss"] = "strategy_specific"
    return normalized


def assert_strategy_configs_compatible(strategy_runs: dict[str, list[dict[str, Any]]]) -> None:
    reference = normalize_strategy_config(strategy_runs["control"][0]["config"])
    for strategy, runs in strategy_runs.items():
        for run in runs:
            if normalize_strategy_config(run["config"]) != reference:
                raise ValueError(f"Strategy config differs beyond allowed fields: {strategy} seed {run['config']['seed']}")


def summarize_strategy(strategy: str, runs: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    metrics = pd.DataFrame([run_metrics(run) for run in runs])
    metrics.insert(0, "strategy", strategy)
    summary = build_metrics_summary(metrics)
    consistency = build_prediction_consistency(runs)
    consistency.insert(0, "strategy", strategy)
    return metrics, summary, consistency


def choose_strategy(summary_frame: pd.DataFrame) -> dict[str, Any]:
    ordered = summary_frame.sort_values("balanced_accuracy_mean", ascending=False).reset_index(drop=True)
    top = ordered.iloc[0]
    close = ordered[ordered["balanced_accuracy_mean"] >= top["balanced_accuracy_mean"] - SELECTION_TOLERANCE].copy()
    close = close.sort_values(
        ["brier_score_mean", "balanced_accuracy_sample_std", "label_disagreement", "all_seeds_incorrect"],
        ascending=[True, True, True, True],
    )
    selected = close.iloc[0]
    control = summary_frame[summary_frame["strategy"] == "control"].iloc[0]
    warnings = []
    if selected["strategy"] != "control":
        sensitivity_drop = control["sensitivity_mean"] - selected["sensitivity_mean"]
        specificity_drop = control["specificity_mean"] - selected["specificity_mean"]
        if sensitivity_drop > 0.01:
            warnings.append(f"sensitivity drops by {sensitivity_drop:.6f} vs control")
        if specificity_drop > 0.01:
            warnings.append(f"specificity drops by {specificity_drop:.6f} vs control")
    conclusion = selected["strategy"]
    if warnings:
        conclusion = "trade_off"
    return {
        "selected": selected["strategy"],
        "conclusion": conclusion,
        "warnings": warnings,
        "balanced_accuracy_tolerance": SELECTION_TOLERANCE,
    }


def build_cross_strategy_consistency(strategy_consistency: dict[str, pd.DataFrame]) -> pd.DataFrame:
    key_columns = ["filename", "patient_id", "true_label"]
    base = strategy_consistency["control"][key_columns].copy()
    if base.duplicated().any():
        raise ValueError("Control validation rows are not unique")
    consensus = {}
    for strategy, frame in strategy_consistency.items():
        seed_columns = [f"predicted_label_seed_{seed}" for seed in EXPECTED_SEEDS]
        prob_columns = [f"probability_seed_{seed}" for seed in EXPECTED_SEEDS]
        columns = key_columns + seed_columns + prob_columns
        candidate = frame[columns].copy()
        if candidate[key_columns].duplicated().any():
            raise ValueError(f"Validation rows are not unique for strategy {strategy}")
        aligned = base.merge(candidate, on=key_columns, how="left", validate="one_to_one")
        if aligned[seed_columns + prob_columns].isna().any().any() or len(aligned) != len(base):
            raise ValueError(f"Validation sample set differs for strategy {strategy}")
        labels = aligned[seed_columns].to_numpy(int)
        consensus[strategy] = (labels.mean(axis=1) >= 0.5).astype(int)
        base[f"{strategy}_mean_probability"] = aligned[prob_columns].mean(axis=1)
        base[f"{strategy}_consensus_label"] = consensus[strategy]
    label_stack = np.column_stack([consensus[strategy] for strategy in STRATEGIES])
    base["strategy_label_disagreement"] = label_stack.max(axis=1) != label_stack.min(axis=1)
    return base


def summary_rows(strategy_summaries: dict[str, dict[str, Any]], strategy_consistency: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for strategy, summary in strategy_summaries.items():
        consistency = strategy_consistency[strategy]
        row = {"strategy": strategy}
        for metric in (*CORE_METRICS, "tn", "fp", "fn", "tp"):
            row[f"{metric}_mean"] = summary[metric]["mean"]
            row[f"{metric}_sample_std"] = summary[metric]["sample_std"]
            row[f"{metric}_min"] = summary[metric]["min"]
            row[f"{metric}_max"] = summary[metric]["max"]
        row["all_seeds_incorrect"] = int(consistency["all_seeds_incorrect"].sum())
        row["label_disagreement"] = int(consistency["label_disagreement"].sum())
        row["std_probability_ge_0.10"] = int((consistency["std_probability"] >= 0.10).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def write_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    metrics = ["balanced_accuracy", "sensitivity", "specificity", "f1"]
    x = np.arange(len(metrics))
    width = 0.22
    for index, row in summary.iterrows():
        axis.bar(x + (index - 1) * width, [row[f"{metric}_mean"] for metric in metrics], width, label=row["strategy"])
    axis.set_ylim(0.85, 1.01)
    axis.set_xticks(x, metrics, rotation=20, ha="right")
    axis.set_ylabel("Validation mean")
    axis.set_title("Validation imbalance strategy comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "strategy_comparison.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 5))
    for row in summary.itertuples(index=False):
        axis.scatter(row.specificity_mean, row.sensitivity_mean, s=80, label=row.strategy)
        axis.annotate(row.strategy, (row.specificity_mean, row.sensitivity_mean), xytext=(5, 5), textcoords="offset points")
    axis.set_xlabel("Validation specificity mean")
    axis.set_ylabel("Validation sensitivity mean")
    axis.set_title("Validation sensitivity-specificity trade-off")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "sensitivity_specificity_tradeoff.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(summary["strategy"], summary["brier_score_mean"], yerr=summary["brier_score_sample_std"], capsize=4)
    axis.set_ylabel("Validation Brier score mean")
    axis.set_title("Validation Brier score by imbalance strategy")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "brier_score_comparison.png", dpi=160)
    plt.close(fig)


def write_report(summary: pd.DataFrame, metrics: pd.DataFrame, selection: dict[str, Any], output_dir: Path) -> None:
    warnings = "; ".join(selection["warnings"]) if selection["warnings"] else "none"
    lines = [
        "# EfficientNetB0 imbalance strategy validation summary", "",
        "## Purpose", "",
        "Compare fixed-threshold validation behavior for control BCE, balanced class weight BCE, and class-balanced focal loss.",
        "The test split was not loaded, predicted, or evaluated.", "",
        "## Strategy definitions", "",
        "- Control: binary crossentropy, no class weight.",
        "- Class weight: binary crossentropy with weights computed from train only: 3821 / (2 * count).",
        "- Focal: BinaryFocalCrossentropy with class balancing; TensorFlow alpha weights class 1 and 1-alpha weights class 0.",
        "- Focal and Keras class_weight are not combined because that would apply two independent imbalance reweighting mechanisms.", "",
        "## Parameter derivation", "",
        "- Train counts: NORMAL=1079, PNEUMONIA=2742, total=3821.",
        "- Class weight NORMAL: 3821 / (2 * 1079) = 1.7706209453.",
        "- Class weight PNEUMONIA: 3821 / (2 * 2742) = 0.6967541940.",
        "- Focal alpha: 1079 / 3821 = 0.2823868097. In TensorFlow this applies to class 1, so PNEUMONIA receives alpha and NORMAL receives 1-alpha=0.7176131903.", "",
        "## Metrics by strategy and seed", "", markdown_table(metrics), "",
        "## Strategy mean summary", "", markdown_table(summary), "",
        "## Selection rule", "",
        "- First priority: highest mean balanced_accuracy.",
        "- If within 0.002, prefer lower mean Brier score.",
        "- Then lower balanced_accuracy sample std, then fewer label disagreements, then fewer all-seeds-incorrect samples.",
        f"- Selected by rule: {selection['selected']}.",
        f"- Final conclusion: {selection['conclusion']}.",
        f"- Safety trade-off warning: {warnings}.", "",
        "## Protocol cautions", "",
        "- BCE val_loss, weighted BCE val_loss, and focal val_loss are on different scales and are not compared across strategies.",
        "- Threshold stayed fixed at 0.5; no threshold optimization was performed.",
        "- No NaN, Inf, OOM, or failed registered runs are accepted by the aggregator.",
        "- Results are validation-only and do not establish final test or clinical performance.", "",
        "## Limitations", "",
        "Only three seeds per strategy are planned. No external validation, confidence intervals, subgroup analysis, or clinical utility analysis is included.",
    ]
    text = "\n".join(lines) + "\n"
    (output_dir / "imbalance_strategy_summary.md").write_text(text, encoding="utf-8")
    (PROJECT_ROOT / "reports" / "efficientnetb0_imbalance_results.md").write_text(text, encoding="utf-8")


def aggregate_strategies(control_registry: str | Path, class_weight_registry: str | Path, focal_registry: str | Path) -> Path:
    registries = {"control": control_registry, "class_weight": class_weight_registry, "focal": focal_registry}
    strategy_runs = {}
    for strategy, registry in registries.items():
        aggregate_transfer_registry(registry)
        _, runs = load_registered_runs(registry)
        strategy_runs[strategy] = runs
    assert_strategy_configs_compatible(strategy_runs)

    metrics_frames = []
    strategy_summaries = {}
    strategy_consistency = {}
    for strategy, runs in strategy_runs.items():
        metrics, summary, consistency = summarize_strategy(strategy, runs)
        metrics_frames.append(metrics)
        strategy_summaries[strategy] = summary
        strategy_consistency[strategy] = consistency
    metrics_by_seed = pd.concat(metrics_frames, ignore_index=True)
    summary = summary_rows(strategy_summaries, strategy_consistency)
    cross_consistency = build_cross_strategy_consistency(strategy_consistency)
    selection = choose_strategy(summary)

    output_dir = PROJECT_ROOT / "results" / "experiments" / "efficientnetb0_imbalance_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_seed.to_csv(output_dir / "strategy_metrics_by_seed.csv", index=False)
    summary.to_csv(output_dir / "strategy_metrics_summary.csv", index=False)
    (output_dir / "strategy_metrics_summary.json").write_text(
        json.dumps({"selection": selection, "strategies": strategy_summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    cross_consistency.to_csv(output_dir / "prediction_consistency_by_strategy.csv", index=False)
    write_plots(summary, output_dir)
    write_report(summary, metrics_by_seed, selection, output_dir)
    required = (
        "strategy_metrics_by_seed.csv", "strategy_metrics_summary.csv", "strategy_metrics_summary.json",
        "prediction_consistency_by_strategy.csv", "strategy_comparison.png",
        "sensitivity_specificity_tradeoff.png", "brier_score_comparison.png",
        "imbalance_strategy_summary.md",
    )
    missing = [name for name in required if not (output_dir / name).is_file() or (output_dir / name).stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing comparison outputs: {missing}")
    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-registry", required=True)
    parser.add_argument("--class-weight-registry", required=True)
    parser.add_argument("--focal-registry", required=True)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    output = aggregate_strategies(args.control_registry, args.class_weight_registry, args.focal_registry)
    print(f"IMBALANCE_SUMMARY_DIR={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
