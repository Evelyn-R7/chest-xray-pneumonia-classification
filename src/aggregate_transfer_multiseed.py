"""Aggregate registered EfficientNetB0 transfer validation-only multiseed runs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SEEDS = (42, 2025, 2026)
EXPECTED_PREDICTIONS = 954
FIXED_THRESHOLD = 0.5
REQUIRED_ARTIFACTS = (
    "resolved_config.yaml", "environment.json", "model_summary.txt",
    "trainable_layers_phase1.txt", "trainable_layers_phase2.txt",
    "phase1_history.csv", "phase2_history.csv", "combined_history.csv",
    "phase1_best.keras", "phase2_best.keras", "best_model.keras",
    "val_predictions.csv", "val_metrics.json", "run_summary.md",
    "learning_curves.png", "confusion_matrix_val.png", "roc_curve_val.png",
    "pr_curve_val.png", "console.log",
)
CORE_METRICS = (
    "accuracy", "precision", "sensitivity", "specificity", "f1",
    "balanced_accuracy", "roc_auc", "pr_auc", "npv", "brier_score",
)
COUNT_METRICS = ("tn", "fp", "fn", "tp")
SUMMARY_METRICS = (
    *CORE_METRICS, "val_loss", "best_epoch", "phase1_epochs", "phase2_epochs",
    "total_training_epochs", "training_duration", *COUNT_METRICS,
)
SUMMARY_ARTIFACTS = (
    "metrics_by_seed.csv", "metrics_summary.json", "val_prediction_consistency.csv",
    "multiseed_metrics.png", "probability_variability.png",
    "confusion_matrices_by_seed.png", "multiseed_summary.md",
)


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_run_summary(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    best_phase = re.search(r"Best phase:\s*(phase[12])", text)
    if not best_phase:
        raise ValueError(f"Best phase not recorded in {path}")
    return {"best_phase": best_phase.group(1)}


def parse_training_duration_seconds(log_path: str | Path) -> float:
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    durations = [float(value) for value in re.findall(r"\d+/\d+\s+-\s+([0-9.]+)s\s+-", text)]
    return float(sum(durations))


def validate_complete_transfer_run(run_dir: str | Path, expected_seed: int | None = None) -> dict[str, Any]:
    run_dir = resolve_project_path(run_dir).resolve()
    missing = [name for name in REQUIRED_ARTIFACTS if not (run_dir / name).is_file() or (run_dir / name).stat().st_size == 0]
    if missing:
        raise ValueError(f"Incomplete transfer run {run_dir}; missing/empty: {missing}")
    config = read_yaml(run_dir / "resolved_config.yaml")
    seed = int(config.get("seed"))
    if expected_seed is not None and seed != expected_seed:
        raise ValueError(f"Expected seed {expected_seed}, found {seed}: {run_dir}")
    if config.get("model_name") != "efficientnetb0":
        raise ValueError(f"Run is not EfficientNetB0: {run_dir}")
    if config.get("data_config") != "configs/data_v3_clean.yaml":
        raise ValueError(f"Run does not use v3_clean: {run_dir}")
    if config.get("run_type") != "full":
        raise ValueError(f"Run is not full: {run_dir}")
    if int(config.get("batch_size")) != 16 or float(config.get("threshold")) != FIXED_THRESHOLD:
        raise ValueError(f"Batch size or threshold changed: {run_dir}")
    if bool(config.get("mixed_precision")):
        raise ValueError(f"Forbidden mixed precision setting: {run_dir}")
    loss = config.get("loss", "binary_crossentropy")
    if loss not in {"binary_crossentropy", "binary_focal_crossentropy"}:
        raise ValueError(f"Unsupported transfer loss: {run_dir}")
    focal = config.get("focal_loss")
    if loss == "binary_focal_crossentropy":
        if not isinstance(focal, dict):
            raise ValueError(f"Focal run is missing focal_loss settings: {run_dir}")
        if config.get("class_weight") is not None and bool(focal.get("apply_class_balancing")):
            raise ValueError(f"Focal class balancing and class_weight cannot both be enabled: {run_dir}")
    elif focal is not None:
        raise ValueError(f"focal_loss settings require focal loss: {run_dir}")
    if int(config.get("fine_tune_last_non_bn_layers")) != 20 or bool(config.get("backbone_training_mode")):
        raise ValueError(f"EfficientNetB0 fine-tune policy changed: {run_dir}")
    if int(config.get("data_num_parallel_calls")) != 1 or int(config.get("augmentation_num_parallel_calls")) != 1:
        raise ValueError(f"WSL serial read setting changed: {run_dir}")
    if bool(config.get("data_prefetch")):
        raise ValueError(f"WSL prefetch setting changed: {run_dir}")

    predictions = pd.read_csv(run_dir / "val_predictions.csv")
    if len(predictions) != EXPECTED_PREDICTIONS:
        raise ValueError(f"Expected 954 validation predictions, found {len(predictions)}: {run_dir}")
    metrics = read_json(run_dir / "val_metrics.json")
    for key in (*CORE_METRICS, *COUNT_METRICS, "threshold"):
        if key not in metrics or not math.isfinite(float(metrics[key])):
            raise ValueError(f"Metric {key!r} is missing or non-finite: {run_dir}")
    if float(metrics["threshold"]) != FIXED_THRESHOLD:
        raise ValueError(f"Metric threshold changed: {run_dir}")
    phase1 = pd.read_csv(run_dir / "phase1_history.csv")
    phase2 = pd.read_csv(run_dir / "phase2_history.csv")
    combined = pd.read_csv(run_dir / "combined_history.csv")
    if phase1.empty or phase2.empty or len(combined) != len(phase1) + len(phase2):
        raise ValueError(f"Both training phases must have non-empty histories: {run_dir}")
    for name, frame in {"phase1": phase1, "phase2": phase2, "combined": combined}.items():
        numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        if numeric.size == 0 or not np.all(np.isfinite(numeric)):
            raise ValueError(f"{name} history contains NaN or Inf: {run_dir}")
    summary = parse_run_summary(run_dir / "run_summary.md")
    if summary["best_phase"] not in {"phase1", "phase2"}:
        raise ValueError(f"Invalid best phase: {run_dir}")
    log = (run_dir / "console.log").read_text(encoding="utf-8", errors="replace")
    for pattern in (r"\bnan\b", r"\binf\b", "out of memory", "resource_exhausted", "traceback", "training failed"):
        if re.search(pattern, log, re.IGNORECASE):
            raise ValueError(f"Suspicious training log pattern {pattern!r}: {run_dir}")
    for path in run_dir.iterdir():
        lower = path.name.lower()
        if "test" in lower and ("prediction" in lower or "metric" in lower):
            raise ValueError(f"Forbidden test output: {path}")
    return {
        "run_dir": run_dir, "config": config, "predictions": predictions, "metrics": metrics,
        "phase1_history": phase1, "phase2_history": phase2, "combined_history": combined,
        "best_phase": summary["best_phase"], "training_duration": parse_training_duration_seconds(run_dir / "console.log"),
    }


def normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    ignored = {"seed", "run_timestamp", "output_dir"}
    return {key: value for key, value in config.items() if key not in ignored}


def assert_configs_equivalent(runs: Iterable[dict[str, Any]]) -> None:
    runs = list(runs)
    reference = normalized_config(runs[0]["config"])
    for run in runs[1:]:
        if normalized_config(run["config"]) != reference:
            raise ValueError("Resolved configurations differ by fields other than seed/time/output path")


def assert_prediction_alignment(frames: list[pd.DataFrame]) -> None:
    reference = frames[0].reset_index(drop=True)
    for frame in frames[1:]:
        candidate = frame.reset_index(drop=True)
        for column in ("filename", "patient_id", "true_label"):
            if not candidate[column].equals(reference[column]):
                raise ValueError(f"Validation {column} order/content differs across seeds")


def summarize_values(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size != 3 or not np.all(np.isfinite(array)):
        raise ValueError("Exactly three finite values are required")
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def build_prediction_consistency(runs: list[dict[str, Any]]) -> pd.DataFrame:
    frames = [run["predictions"] for run in runs]
    assert_prediction_alignment(frames)
    seeds = [int(run["config"]["seed"]) for run in runs]
    probabilities = np.column_stack([frame["predicted_probability"].to_numpy(float) for frame in frames])
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Prediction probabilities contain NaN or Inf")
    true_labels = frames[0]["true_label"].to_numpy(int)
    predicted = (probabilities >= FIXED_THRESHOLD).astype(int)
    incorrect = (predicted != true_labels[:, None]).sum(axis=1)
    consistency = pd.DataFrame({
        "filename": frames[0]["filename"].to_numpy(),
        "patient_id": frames[0]["patient_id"].to_numpy(),
        "true_label": true_labels,
    })
    for index, seed in enumerate(seeds):
        consistency[f"probability_seed_{seed}"] = probabilities[:, index]
    for index, seed in enumerate(seeds):
        consistency[f"predicted_label_seed_{seed}"] = predicted[:, index]
    consistency["mean_probability"] = probabilities.mean(axis=1)
    consistency["std_probability"] = probabilities.std(axis=1, ddof=1)
    consistency["incorrect_count"] = incorrect
    consistency["label_disagreement"] = predicted.max(axis=1) != predicted.min(axis=1)
    consistency["all_seeds_correct"] = incorrect == 0
    consistency["all_seeds_incorrect"] = incorrect == len(runs)
    return consistency


def run_metrics(run: dict[str, Any]) -> dict[str, Any]:
    phase1 = run["phase1_history"]
    phase2 = run["phase2_history"]
    combined = run["combined_history"]
    best_position = int(combined["val_loss"].astype(float).argmin())
    best_epoch = int(combined.iloc[best_position].get("epoch", best_position)) + 1
    row: dict[str, Any] = {"seed": int(run["config"]["seed"]), "best_phase": run["best_phase"]}
    row.update({key: float(run["metrics"][key]) for key in CORE_METRICS})
    row.update({key: int(run["metrics"][key]) for key in COUNT_METRICS})
    row["val_loss"] = float(combined.iloc[best_position]["val_loss"])
    row["best_epoch"] = best_epoch
    row["phase1_epochs"] = len(phase1)
    row["phase2_epochs"] = len(phase2)
    row["total_training_epochs"] = len(combined)
    row["training_duration"] = float(run["training_duration"])
    return row


def build_metrics_summary(metrics_by_seed: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in SUMMARY_METRICS:
        values = metrics_by_seed[key].astype(float).tolist()
        summary[key] = {f"seed_{seed}": value for seed, value in zip(EXPECTED_SEEDS, values)}
        summary[key].update(summarize_values(values))
    summary["best_phase"] = {f"seed_{int(row.seed)}": row.best_phase for row in metrics_by_seed.itertuples()}
    return summary


def write_plots(metrics: pd.DataFrame, consistency: pd.DataFrame, output_dir: Path) -> None:
    keys = ["accuracy", "sensitivity", "specificity", "balanced_accuracy", "roc_auc", "pr_auc"]
    x = np.arange(len(keys))
    width = 0.22
    fig, axis = plt.subplots(figsize=(11, 6))
    for index, row in metrics.iterrows():
        axis.bar(x + (index - 1) * width, [row[key] for key in keys], width, label=f"seed {int(row['seed'])}")
    axis.plot(x, metrics[keys].mean(axis=0), "ko--", label="mean")
    axis.set_xticks(x, keys, rotation=25, ha="right")
    axis.set_ylim(0.85, 1.01)
    axis.set_ylabel("Validation metric")
    axis.set_title("EfficientNetB0 validation metrics across seeds")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "multiseed_metrics.png", dpi=160)
    plt.close(fig)

    high_variability = int((consistency["std_probability"] >= 0.10).sum())
    disagreements = int(consistency["label_disagreement"].sum())
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(consistency["std_probability"], bins="auto", edgecolor="black", alpha=0.8)
    axis.axvline(0.10, color="tab:red", linestyle="--", label="std_probability = 0.10")
    axis.set_xlabel("Validation probability sample standard deviation")
    axis.set_ylabel("Sample count")
    axis.set_title(
        f"Validation probability variability\nstd >= 0.10: {high_variability}; label disagreements: {disagreements}"
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "probability_variability.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharex=True, sharey=True)
    for axis, row in zip(axes, metrics.itertuples(index=False)):
        matrix = np.asarray([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"Validation seed {int(row.seed)}")
        axis.set_xticks([0, 1], ["Pred 0", "Pred 1"])
        axis.set_yticks([0, 1], ["True 0", "True 1"])
        for y in range(2):
            for x_pos in range(2):
                axis.text(x_pos, y, str(matrix[y, x_pos]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrices_by_seed.png", dpi=160)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def write_summary_markdown(
    metrics: pd.DataFrame,
    summary: dict[str, Any],
    consistency: pd.DataFrame,
    output_dir: Path,
    registry: dict[str, Any],
) -> None:
    stds = {key: summary[key]["sample_std"] for key in CORE_METRICS}
    stable = min(stds, key=stds.get)
    variable = max(stds, key=stds.get)
    all_wrong = int(consistency["all_seeds_incorrect"].sum())
    disagreements = int(consistency["label_disagreement"].sum())
    high_probability_std = int((consistency["std_probability"] >= 0.10).sum())
    degenerate = []
    for row in metrics.itertuples(index=False):
        if row.tp + row.fp == EXPECTED_PREDICTIONS:
            degenerate.append(f"seed {int(row.seed)} all positive")
        if row.tn + row.fn == EXPECTED_PREDICTIONS:
            degenerate.append(f"seed {int(row.seed)} all negative")
    degenerate_text = "; ".join(degenerate) if degenerate else "none"
    manifest_rows = pd.DataFrame(
        [{"manifest": path, "sha256": digest} for path, digest in registry.get("manifest_sha256", {}).items()]
    )
    cnn_comparison = pd.DataFrame([
        {"metric": "accuracy", "CNN baseline mean ± std": "0.892383 ± 0.019815", "EfficientNetB0 mean ± std": f"{summary['accuracy']['mean']:.6f} ± {summary['accuracy']['sample_std']:.6f}"},
        {"metric": "sensitivity", "CNN baseline mean ± std": "0.952242 ± 0.060797", "EfficientNetB0 mean ± std": f"{summary['sensitivity']['mean']:.6f} ± {summary['sensitivity']['sample_std']:.6f}"},
        {"metric": "specificity", "CNN baseline mean ± std": "0.740741 ± 0.213534", "EfficientNetB0 mean ± std": f"{summary['specificity']['mean']:.6f} ± {summary['specificity']['sample_std']:.6f}"},
        {"metric": "f1", "CNN baseline mean ± std": "0.927242 ± 0.009169", "EfficientNetB0 mean ± std": f"{summary['f1']['mean']:.6f} ± {summary['f1']['sample_std']:.6f}"},
        {"metric": "roc_auc", "CNN baseline mean ± std": "0.974347 ± 0.006318", "EfficientNetB0 mean ± std": f"{summary['roc_auc']['mean']:.6f} ± {summary['roc_auc']['sample_std']:.6f}"},
        {"metric": "pr_auc", "CNN baseline mean ± std": "0.989708 ± 0.002901", "EfficientNetB0 mean ± std": f"{summary['pr_auc']['mean']:.6f} ± {summary['pr_auc']['sample_std']:.6f}"},
        {"metric": "brier_score", "CNN baseline mean ± std": "0.080924 ± 0.015574", "EfficientNetB0 mean ± std": f"{summary['brier_score']['mean']:.6f} ± {summary['brier_score']['sample_std']:.6f}"},
    ])
    report = [
        "# EfficientNetB0 transfer multiseed validation summary", "",
        "All values are validation-only. The test split was not loaded, predicted, or evaluated.", "",
        "## Manifest SHA-256", "",
        markdown_table(manifest_rows), "",
        "## Metrics by seed", "", markdown_table(metrics), "", "## Mean ± sample std", "",
    ]
    for key in SUMMARY_METRICS:
        report.append(f"- {key}: {summary[key]['mean']:.6f} ± {summary[key]['sample_std']:.6f}")
    report += [
        "", "## Stability checks", "",
        f"- Most stable metric: {stable}",
        f"- Most variable metric: {variable}",
        f"- Samples incorrect for all three seeds: {all_wrong}",
        f"- Samples with prediction label disagreement: {disagreements}",
        f"- Samples with probability sample std >= 0.10: {high_probability_std}",
        f"- Degenerate all-positive or all-negative runs: {degenerate_text}",
        "- NaN, Inf, OOM, or failed registered runs: none detected by aggregation checks",
        "- Threshold remained fixed at 0.5; no threshold tuning was performed", "",
        "## Descriptive comparison to CNN baseline", "",
        "This report is descriptive and does not claim statistical superiority without formal testing.",
        "CNN baseline numbers are copied from `reports/cnn_baseline_multiseed_results.md` for validation-only context.",
        "", markdown_table(cnn_comparison),
        "", "## Limitations", "",
        "Only three seeds are summarized. Results are validation-only and do not establish final test or clinical performance.",
        "No external validation, confidence intervals, subgroup analysis, calibration study, or clinical utility analysis is included.",
    ]
    text = "\n".join(report) + "\n"
    (output_dir / "multiseed_summary.md").write_text(text, encoding="utf-8")
    (PROJECT_ROOT / "reports" / "efficientnetb0_multiseed_results.md").write_text(text, encoding="utf-8")


def load_registered_runs(registry_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    registry = read_json(registry_path)
    if registry.get("model_name") != "efficientnetb0" or float(registry.get("fixed_threshold")) != FIXED_THRESHOLD:
        raise ValueError("Registry is not for EfficientNetB0 fixed-threshold transfer aggregation")
    runs = []
    for seed in EXPECTED_SEEDS:
        key = f"seed_{seed}_run_dir"
        value = registry.get(key)
        if not value:
            raise ValueError(f"Registry is missing concrete path for {key}")
        runs.append(validate_complete_transfer_run(value, expected_seed=seed))
    return registry, runs


def aggregate_transfer_registry(registry_path: str | Path) -> Path:
    registry_path = resolve_project_path(registry_path).resolve()
    registry, runs = load_registered_runs(registry_path)
    assert_configs_equivalent(runs)
    consistency = build_prediction_consistency(runs)
    metrics = pd.DataFrame([run_metrics(run) for run in runs])
    summary = build_metrics_summary(metrics)
    output_dir = registry_path.parent / "multiseed_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "metrics_by_seed.csv", index=False)
    consistency.to_csv(output_dir / "val_prediction_consistency.csv", index=False)
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_plots(metrics, consistency, output_dir)
    write_summary_markdown(metrics, summary, consistency, output_dir, registry)
    missing = [name for name in SUMMARY_ARTIFACTS if not (output_dir / name).is_file() or (output_dir / name).stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Summary generation incomplete; missing/empty: {missing}")
    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    output = aggregate_transfer_registry(args.registry)
    print(f"SUMMARY_DIR={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
