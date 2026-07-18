"""Validate and aggregate registered CNN baseline validation-only multiseed runs."""

from __future__ import annotations

import argparse
import json
import math
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
    "resolved_config.yaml", "environment.json", "model_summary.txt", "history.csv",
    "best_model.keras", "val_predictions.csv", "val_metrics.json", "run_summary.md",
    "learning_curves.png", "confusion_matrix_val.png", "roc_curve_val.png", "pr_curve_val.png",
)
CORE_METRICS = (
    "accuracy", "precision", "sensitivity", "specificity", "f1", "balanced_accuracy",
    "roc_auc", "pr_auc", "npv", "brier_score",
)
SUMMARY_ARTIFACTS = (
    "metrics_by_seed.csv", "metrics_summary.json", "val_prediction_consistency.csv",
    "multiseed_metrics.png", "probability_variability.png", "multiseed_summary.md",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def validate_complete_run(
    run_dir: str | Path, expected_seed: int | None = None,
    expected_predictions: int = EXPECTED_PREDICTIONS,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    missing = [name for name in REQUIRED_ARTIFACTS if not (run_dir / name).is_file() or (run_dir / name).stat().st_size == 0]
    if missing:
        raise ValueError(f"Incomplete run {run_dir}; missing/empty: {missing}")
    config = _read_yaml(run_dir / "resolved_config.yaml")
    if config.get("run_type") != "full":
        raise ValueError(f"Run is not full: {run_dir}")
    seed = int(config.get("seed"))
    if expected_seed is not None and seed != expected_seed:
        raise ValueError(f"Expected seed {expected_seed}, found {seed}: {run_dir}")
    if float(config.get("threshold")) != FIXED_THRESHOLD:
        raise ValueError(f"Threshold is not fixed at 0.5: {run_dir}")
    if config.get("data_config") != "configs/data_v3_clean.yaml":
        raise ValueError(f"Run does not use v3_clean data config: {run_dir}")

    predictions = pd.read_csv(run_dir / "val_predictions.csv")
    if len(predictions) != expected_predictions:
        raise ValueError(f"Expected {expected_predictions} validation predictions, found {len(predictions)}: {run_dir}")
    metrics = _read_json(run_dir / "val_metrics.json")
    for key in CORE_METRICS:
        if key not in metrics or not math.isfinite(float(metrics[key])):
            raise ValueError(f"Metric {key!r} is missing or non-finite: {run_dir}")
    if float(metrics.get("threshold", FIXED_THRESHOLD)) != FIXED_THRESHOLD:
        raise ValueError(f"Metric threshold is not 0.5: {run_dir}")
    history = pd.read_csv(run_dir / "history.csv")
    if history.empty or "val_loss" not in history:
        raise ValueError(f"Training history is empty or lacks val_loss: {run_dir}")
    numeric = history.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"Training history contains NaN or Inf: {run_dir}")
    return {"run_dir": run_dir, "config": config, "predictions": predictions, "metrics": metrics, "history": history}


def find_unique_seed42_full(seed_root: str | Path) -> Path:
    seed_root = Path(seed_root).resolve()
    candidates = []
    for config_path in sorted(seed_root.rglob("resolved_config.yaml")):
        try:
            validated = validate_complete_run(config_path.parent, expected_seed=42)
        except (ValueError, OSError, json.JSONDecodeError, pd.errors.ParserError):
            continue
        candidates.append(validated["run_dir"])
    if len(candidates) != 1:
        listing = "\n".join(str(path) for path in candidates) or "(none)"
        raise RuntimeError(f"Expected exactly one complete seed 42 full run; found {len(candidates)}:\n{listing}")
    return candidates[0]


def normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    ignored = {"seed", "run_timestamp", "output_dir"}
    return {key: value for key, value in config.items() if key not in ignored}


def assert_configs_equivalent(runs: Iterable[dict[str, Any]]) -> None:
    runs = list(runs)
    reference = normalized_config(runs[0]["config"])
    for run in runs[1:]:
        if normalized_config(run["config"]) != reference:
            raise ValueError("Resolved configurations differ by fields other than seed/time/output path")


def _portable_filepath(value: str) -> str:
    text = str(value).replace("\\", "/")
    marker = "/data/"
    if marker in text:
        return "data/" + text.split(marker, 1)[1]
    if text.startswith("data/"):
        return text
    raise ValueError(f"Cannot convert filepath to a project-relative path: {value}")


def assert_prediction_alignment(frames: list[pd.DataFrame]) -> list[str]:
    if any(len(frame) != EXPECTED_PREDICTIONS for frame in frames):
        raise ValueError("Every run must contain exactly 954 validation predictions")
    portable = [[_portable_filepath(value) for value in frame["filepath"]] for frame in frames]
    for index in range(1, len(frames)):
        if portable[index] != portable[0]:
            raise ValueError("Validation filepath order/content differs across seeds")
        for column in ("patient_id", "true_label", "filename"):
            if not frames[index][column].reset_index(drop=True).equals(frames[0][column].reset_index(drop=True)):
                raise ValueError(f"Validation {column} order/content differs across seeds")
    return portable[0]


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
    paths = assert_prediction_alignment(frames)
    probabilities = np.column_stack([frame["predicted_probability"].to_numpy(float) for frame in frames])
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Prediction probabilities must be finite and within [0, 1]")
    true_labels = frames[0]["true_label"].to_numpy(int)
    predicted = (probabilities >= FIXED_THRESHOLD).astype(int)
    incorrect = (predicted != true_labels[:, None]).sum(axis=1)
    seeds = [int(run["config"]["seed"]) for run in runs]
    result = pd.DataFrame({
        "filepath": paths,
        "filename": frames[0]["filename"].to_numpy(),
        "patient_id": frames[0]["patient_id"].to_numpy(),
        "true_label": true_labels,
    })
    for column, seed in enumerate(seeds):
        result[f"probability_seed_{seed}"] = probabilities[:, column]
    result["mean_probability"] = probabilities.mean(axis=1)
    result["std_probability"] = probabilities.std(axis=1, ddof=1)
    result["incorrect_count"] = incorrect
    result["all_seeds_correct"] = incorrect == 0
    result["all_seeds_incorrect"] = incorrect == len(runs)
    return result


def _run_metrics(run: dict[str, Any]) -> dict[str, Any]:
    history = run["history"]
    best_position = int(history["val_loss"].astype(float).argmin())
    row = {"seed": int(run["config"]["seed"])}
    row.update({key: float(run["metrics"][key]) for key in CORE_METRICS})
    row["val_loss"] = float(history.iloc[best_position]["val_loss"])
    row["best_epoch"] = best_position + 1
    row["training_epochs"] = len(history)
    row["training_duration"] = None
    return row


def _write_plots(metrics: pd.DataFrame, consistency: pd.DataFrame, output_dir: Path) -> None:
    keys = ["accuracy", "sensitivity", "specificity", "balanced_accuracy", "roc_auc", "pr_auc"]
    x = np.arange(len(keys))
    width = 0.2
    fig, axis = plt.subplots(figsize=(12, 6))
    for index, row in metrics.iterrows():
        axis.bar(x + (index - 1) * width, [row[key] for key in keys], width, label=f"seed {int(row['seed'])}")
    axis.plot(x, metrics[keys].mean(axis=0), "ko--", label="mean")
    axis.set_xticks(x, keys, rotation=25, ha="right")
    axis.set_ylabel("Validation metric")
    axis.set_title("CNN baseline validation metrics across seeds")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "multiseed_metrics.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(consistency["std_probability"], bins="auto", edgecolor="black", alpha=0.8)
    axis.set_xlabel("Prediction probability sample standard deviation")
    axis.set_ylabel("Validation sample count")
    axis.set_title("Validation probability variability across seeds")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "probability_variability.png", dpi=160)
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])

def aggregate_registry(registry_path: str | Path, repair_existing: bool = False) -> Path:
    registry_path = Path(registry_path).resolve()
    registry = _read_json(registry_path)
    run_dirs = [Path(registry[f"seed_{seed}_run_dir"]) for seed in EXPECTED_SEEDS]
    runs = [validate_complete_run(path, expected_seed=seed) for path, seed in zip(run_dirs, EXPECTED_SEEDS)]
    assert_configs_equivalent(runs)
    consistency = build_prediction_consistency(runs)
    rows = [_run_metrics(run) for run in runs]
    metrics = pd.DataFrame(rows)

    output_dir = registry_path.parent / "multiseed_summary"
    if repair_existing and not output_dir.is_dir():
        raise FileNotFoundError(f"Cannot repair missing summary directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=repair_existing)
    metrics.to_csv(output_dir / "metrics_by_seed.csv", index=False)
    consistency.to_csv(output_dir / "val_prediction_consistency.csv", index=False)
    summary = {}
    for key in (*CORE_METRICS, "val_loss", "best_epoch", "training_epochs"):
        values = metrics[key].astype(float).tolist()
        summary[key] = {f"seed_{seed}": value for seed, value in zip(EXPECTED_SEEDS, values)}
        summary[key].update(summarize_values(values))
    summary["training_duration"] = {f"seed_{seed}": None for seed in EXPECTED_SEEDS}
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_plots(metrics, consistency, output_dir)

    stds = {key: summary[key]["sample_std"] for key in CORE_METRICS}
    stable = min(stds, key=stds.get)
    variable = max(stds, key=stds.get)
    all_wrong = int(consistency["all_seeds_incorrect"].sum())
    high_variability = int((consistency["std_probability"] >= 0.10).sum())
    degenerate = []
    for run in runs:
        predicted = (run["predictions"]["predicted_probability"].to_numpy(float) >= FIXED_THRESHOLD).astype(int)
        if np.all(predicted == 1):
            degenerate.append(f"seed {run['config']['seed']} 全部预测阳性")
        elif np.all(predicted == 0):
            degenerate.append(f"seed {run['config']['seed']} 全部预测阴性")
    degenerate_text = "；".join(degenerate) if degenerate else "无"

    lines = [
        "# CNN baseline 多随机种子验证结果", "",
        "单一随机种子可能偶然受初始化、shuffle 和随机增强影响，不能充分证明稳定性。三个运行除随机种子、时间戳和输出目录外配置一致。", "",
        "## 逐种子指标", "", _markdown_table(metrics), "", "## 均值与样本标准差", "",
    ]
    for key in (*CORE_METRICS, "val_loss", "best_epoch", "training_epochs"):
        lines.append(f"- {key}: {summary[key]['mean']:.6f} ± {summary[key]['sample_std']:.6f}")
    lines += [
        "", "## 稳定性", "", f"- 最稳定指标：{stable}（最小跨 seed 样本标准差）",
        f"- 波动最大指标：{variable}（最大跨 seed 样本标准差）",
        f"- 三个 seed 均预测错误的验证样本：{all_wrong}",
        f"- 高概率波动样本：{high_variability}（定义：三个 seed 概率的样本标准差 ≥ 0.10）",
        f"- 全阳性或全阴性运行：{degenerate_text}",
        "- NaN、Inf 或训练失败：未发现（登记前已验证 history 与指标均为有限数值）", "",
        "## 协议声明", "",
        "- 固定阈值为 0.5，未进行阈值优化。",
        "- 未加载、预测或评估 test 集；所有数值均来自 954 张 validation 图像。",
        "- manifest SHA-256 由登记文件记录并在训练脚本前后核对。",
        "- 本分析不用于反向修改模型或阈值。", "",
        "## 局限性", "",
        "仅三个随机种子仍不足以完整刻画不确定性；结果来自单一内部验证划分，未提供外部验证、置信区间、亚组分析或临床效用证明。",
    ]
    report_text = "\n".join(lines) + "\n"
    (output_dir / "multiseed_summary.md").write_text(report_text, encoding="utf-8")
    (PROJECT_ROOT / "reports" / "cnn_baseline_multiseed_results.md").write_text(report_text, encoding="utf-8")
    missing = [name for name in SUMMARY_ARTIFACTS if not (output_dir / name).is_file() or (output_dir / name).stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Summary generation incomplete; missing/empty: {missing}")
    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument(
        "--repair-existing", action="store_true",
        help="Regenerate an existing summary directory from the same registered validation outputs",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    output = aggregate_registry(args.registry, repair_existing=args.repair_existing)
    print(f"SUMMARY_DIR={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
