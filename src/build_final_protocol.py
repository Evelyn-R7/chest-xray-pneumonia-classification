"""Build and freeze the validation-only final model selection protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.final_protocol.calibration import (
    apply_calibrator,
    build_oof_calibration_candidates,
    calibration_metrics,
    choose_calibration_method,
    fit_calibrator,
)
from src.final_protocol.protocol import (
    EXPECTED_SEEDS,
    FIXED_THRESHOLD,
    MANIFESTS,
    PROJECT_ROOT,
    build_ensemble_predictions,
    load_control_runs,
    metrics_with_log_loss,
    read_json,
    sha256_file,
    utc_now_iso,
    write_json,
)
from src.final_protocol.thresholds import select_balanced_threshold

REGISTRY = PROJECT_ROOT / "results/experiments/efficientnetb0_transfer_v1/multiseed_registry.json"
OUTPUT_DIR = PROJECT_ROOT / "results/final_protocol"
REPORT_PATH = PROJECT_ROOT / "reports/final_model_selection_protocol.md"


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small dataframe as GitHub-flavored Markdown without tabulate."""

    data = frame.copy()
    columns = [str(column) for column in data.columns]

    def format_cell(value) -> str:
        if isinstance(value, float):
            return f"{value:.12g}"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in data.iterrows():
        lines.append("| " + " | ".join(format_cell(row[column]) for column in data.columns) + " |")
    return "\n".join(lines)


def write_calibration_curve(y_true: np.ndarray, probabilities: np.ndarray, output_path: Path) -> None:
    bins = np.linspace(0.0, 1.0, 16)
    xs: list[float] = []
    ys: list[float] = []
    counts: list[int] = []
    for index in range(15):
        left = bins[index]
        right = bins[index + 1]
        mask = (probabilities >= left) & ((probabilities <= right) if index == 14 else (probabilities < right))
        if not np.any(mask):
            continue
        xs.append(float(np.mean(probabilities[mask])))
        ys.append(float(np.mean(y_true[mask])))
        counts.append(int(mask.sum()))
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    axis.scatter(xs, ys, s=[max(30, count * 2) for count in counts], alpha=0.75, label="validation bins")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed positive fraction")
    axis.set_title("Validation calibration curve (15 equal-width bins)")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_threshold_tradeoff(threshold_frame: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    ordered = threshold_frame.sort_values("threshold")
    axis.plot(ordered["threshold"], ordered["sensitivity"], label="sensitivity")
    axis.plot(ordered["threshold"], ordered["specificity"], label="specificity")
    axis.plot(ordered["threshold"], ordered["balanced_accuracy"], label="balanced accuracy")
    axis.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="benchmark 0.5")
    axis.set_xlabel("Threshold")
    axis.set_ylabel("Validation metric")
    axis.set_title("Validation threshold trade-off")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_summary_markdown(
    output_path: Path,
    raw_metrics: dict,
    calibration_summary: dict,
    selected_method: str,
    benchmark_metrics: dict,
    balanced_threshold: float,
    balanced_metrics: dict,
) -> None:
    lines = [
        "# Final protocol summary",
        "",
        "Validation-only Stage 7A output. Test data was not loaded or evaluated.",
        "",
        "## Raw three-seed ensemble at threshold 0.5",
        "",
        markdown_table(pd.DataFrame([raw_metrics])),
        "",
        "## Calibration OOF summary",
        "",
        markdown_table(pd.DataFrame.from_dict(calibration_summary, orient="index").reset_index(names="method")),
        "",
        f"Selected calibration method: `{selected_method}`.",
        "",
        "## Pre-registered thresholds",
        "",
        f"- benchmark_threshold: `{FIXED_THRESHOLD}`",
        f"- balanced_threshold: `{balanced_threshold}`",
        "",
        "## Benchmark threshold metrics",
        "",
        markdown_table(pd.DataFrame([benchmark_metrics])),
        "",
        "## Balanced threshold metrics",
        "",
        markdown_table(pd.DataFrame([balanced_metrics])),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_report(path: Path, calibration_summary: dict, selected_method: str, balanced_threshold: float) -> None:
    class_weight = {
        "balanced_accuracy": 0.963434048083171,
        "brier_score": 0.025693672364964736,
        "sensitivity": 0.9663742690058479,
        "specificity": 0.9604938271604939,
    }
    focal = {
        "balanced_accuracy": 0.963385315139701,
        "brier_score": 0.034186916729747066,
        "sensitivity": 0.9551656920077973,
        "specificity": 0.9716049382716049,
    }
    control = {
        "balanced_accuracy": 0.9691682910981156,
        "brier_score": 0.020623489682667372,
        "sensitivity": 0.9766081871345028,
        "specificity": 0.9617283950617285,
    }
    text = f"""# Final model selection protocol

Stage 7A freezes the final validation-selected protocol before any final test evaluation.

## Frozen model form

The final predictor is the EfficientNetB0 control three-seed equal-probability ensemble:

`ensemble_raw_probability = (p_seed_42 + p_seed_2025 + p_seed_2026) / 3`

Seeds are fixed to 42, 2025, and 2026. No single seed is selected as the final model because single-seed selection would overfit model choice to validation noise. Equal averaging keeps the protocol simple and pre-specified.

## Why control was selected

Stage 6 compared validation-only imbalance strategies at threshold 0.5:

| strategy | mean balanced accuracy | mean sensitivity | mean specificity | mean Brier |
| --- | ---: | ---: | ---: | ---: |
| control | {control['balanced_accuracy']:.9f} | {control['sensitivity']:.9f} | {control['specificity']:.9f} | {control['brier_score']:.9f} |
| class weight | {class_weight['balanced_accuracy']:.9f} | {class_weight['sensitivity']:.9f} | {class_weight['specificity']:.9f} | {class_weight['brier_score']:.9f} |
| focal | {focal['balanced_accuracy']:.9f} | {focal['sensitivity']:.9f} | {focal['specificity']:.9f} | {focal['brier_score']:.9f} |

Control had the highest mean balanced accuracy and best mean Brier score among the three strategies. Focal improved specificity but reduced sensitivity and calibration quality; class weighting did not improve the validation trade-off.

## Calibration methods

Three candidates were compared using validation-only out-of-fold predictions:

1. `none`: no calibration, raw ensemble probability is used.
2. `platt`: logistic regression on the raw ensemble probability.
3. `isotonic`: isotonic regression with `out_of_bounds="clip"`.

The comparison uses `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped by `patient_id`, so the same patient cannot appear in both calibration train and fold validation.

ECE is computed with 15 equal-width bins over [0, 1]. For every non-empty bin, the absolute difference between mean predicted probability and observed positive fraction is weighted by the bin sample fraction; ECE is the weighted sum.

## Calibration selection rule

The pre-registered rule is: choose the lowest OOF Brier score. If methods are within 0.001 Brier, choose the simpler method in this order: none, Platt, isotonic.

Selected method: `{selected_method}`.

OOF summary:

{markdown_table(pd.DataFrame.from_dict(calibration_summary, orient='index').reset_index(names='method'))}

## Thresholds

Two thresholds are pre-registered:

- benchmark_threshold = 0.5
- balanced_threshold = {balanced_threshold}

The balanced threshold is selected on validation probabilities from the selected calibration method by maximizing balanced accuracy. Ties are resolved by higher sensitivity, then distance closer to 0.5, then smaller threshold.

## Protocol cautions

All results in this report are validation-only. Test data has not been loaded, predicted, or evaluated. The final test evaluation should be run only once after this protocol is frozen, and test results must not trigger retraining, re-selection of calibration, or threshold changes.

This model is not clinically deployable. The dataset is limited, patient IDs are inferred from filenames, and there is no external validation, subgroup analysis, prospective validation, or clinical utility analysis.
"""
    path.write_text(text, encoding="utf-8")


def freeze_protocol(output_dir: Path) -> None:
    protocol_path = output_dir / "final_protocol.json"
    frozen_path = output_dir / "PROTOCOL_FROZEN"
    if frozen_path.exists():
        raise FileExistsError(f"Frozen protocol already exists and will not be overwritten: {frozen_path}")
    digest = sha256_file(protocol_path)
    frozen_path.write_text(
        "\n".join([
            f"final_protocol_json_sha256: {digest}",
            f"frozen_at: {utc_now_iso()}",
            "This protocol is frozen. The next stage must not modify the model, calibration method, or thresholds.",
            "After final test evaluation, do not retrain or re-select strategies based on test results.",
            "",
        ]),
        encoding="utf-8",
    )


def build_final_protocol(registry_path: Path = REGISTRY, output_dir: Path = OUTPUT_DIR) -> dict:
    frozen_path = output_dir / "PROTOCOL_FROZEN"
    if frozen_path.exists():
        raise FileExistsError(f"Frozen protocol already exists and will not be overwritten: {frozen_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_control_runs(registry_path)
    ensemble = build_ensemble_predictions(runs)
    ensemble_path = output_dir / "ensemble_validation_predictions.csv"
    ensemble.to_csv(ensemble_path, index=False)

    y_true = ensemble["true_label"].to_numpy(int)
    raw_prob = ensemble["ensemble_raw_probability"].to_numpy(float)
    patient_ids = ensemble["patient_id"].to_numpy()
    raw_metrics = metrics_with_log_loss(y_true, raw_prob, threshold=FIXED_THRESHOLD)

    oof_predictions, fold_metrics, calibration_summary = build_oof_calibration_candidates(y_true, raw_prob, patient_ids)
    for method in ("none", "platt", "isotonic"):
        ensemble[f"oof_probability_{method}"] = oof_predictions[f"oof_probability_{method}"]
    ensemble.to_csv(output_dir / "calibration_oof_predictions.csv", index=False)
    fold_metrics.to_csv(output_dir / "calibration_metrics_by_fold.csv", index=False)
    selected_method = choose_calibration_method(calibration_summary)

    calibrator_path = None
    calibrator_sha = None
    if selected_method != "none":
        calibrator = fit_calibrator(selected_method, raw_prob, y_true)
        calibrated_prob = apply_calibrator(selected_method, calibrator, raw_prob)
        calibrator_path = output_dir / ("platt_calibrator.joblib" if selected_method == "platt" else "isotonic_calibrator.joblib")
        joblib.dump(calibrator, calibrator_path)
        calibrator_sha = sha256_file(calibrator_path)
    else:
        calibrated_prob = raw_prob.copy()

    benchmark_metrics = metrics_with_log_loss(y_true, calibrated_prob, threshold=FIXED_THRESHOLD)
    balanced_threshold, threshold_frame = select_balanced_threshold(y_true, calibrated_prob)
    threshold_frame.to_csv(output_dir / "threshold_candidates.csv", index=False)
    balanced_metrics = metrics_with_log_loss(y_true, calibrated_prob, threshold=balanced_threshold)

    final_metrics = {
        "raw_ensemble_threshold_0_5": raw_metrics,
        "selected_calibration_method": selected_method,
        "selected_calibration_validation_metrics": calibration_metrics(y_true, calibrated_prob),
        "benchmark_threshold": FIXED_THRESHOLD,
        "benchmark_threshold_metrics": benchmark_metrics,
        "balanced_threshold": balanced_threshold,
        "balanced_threshold_metrics": balanced_metrics,
    }
    write_json(output_dir / "validation_metrics_final_protocol.json", final_metrics)

    calibration_summary_payload = {
        "selection_rule": "Lowest OOF Brier score; if Brier difference <= 0.001, prefer simpler method in order none, Platt, isotonic.",
        "ece_definition": "15 equal-width bins over [0, 1]; weighted mean absolute difference between bin confidence and observed positive fraction.",
        "methods": calibration_summary,
        "selected_method": selected_method,
    }
    write_json(output_dir / "calibration_summary.json", calibration_summary_payload)

    write_calibration_curve(y_true, calibrated_prob, output_dir / "calibration_curve_validation.png")
    write_threshold_tradeoff(threshold_frame, output_dir / "threshold_tradeoff_validation.png")
    write_summary_markdown(
        output_dir / "final_protocol_summary.md",
        raw_metrics,
        calibration_summary,
        selected_method,
        benchmark_metrics,
        balanced_threshold,
        balanced_metrics,
    )
    write_report(REPORT_PATH, calibration_summary, selected_method, balanced_threshold)

    manifest_sha = {name: sha256_file(PROJECT_ROOT / name) for name in MANIFESTS}
    val_prediction_sha = {
        f"seed_{run['seed']}": sha256_file(run["predictions_path"])
        for run in runs
    }
    model_sha = {
        f"seed_{run['seed']}": sha256_file(run["model_path"])
        for run in runs
    }
    model_paths = {f"seed_{run['seed']}": str(run["model_path"]) for run in runs}
    run_dirs = {f"seed_{run['seed']}": str(run["run_dir"]) for run in runs}
    protocol = {
        "protocol_version": "stage_7a_v1",
        "created_at": utc_now_iso(),
        "final_model_type": "EfficientNetB0 control three-seed probability ensemble",
        "ensemble_method": "equal arithmetic mean of validation probabilities from seeds 42, 2025, and 2026",
        "seeds": list(EXPECTED_SEEDS),
        "model_run_dirs": run_dirs,
        "best_model_paths": model_paths,
        "best_model_sha256": model_sha,
        "registry_path": str(registry_path.resolve()),
        "registry_sha256": sha256_file(registry_path),
        "val_predictions_sha256": val_prediction_sha,
        "manifest_sha256": manifest_sha,
        "data_protocol": "v3_clean_train_val_only",
        "loss": "binary_crossentropy",
        "class_weight": None,
        "selected_calibration_method": selected_method,
        "calibration_selection_rule": calibration_summary_payload["selection_rule"],
        "calibrator_path": str(calibrator_path) if calibrator_path else None,
        "calibrator_sha256": calibrator_sha,
        "benchmark_threshold": FIXED_THRESHOLD,
        "balanced_threshold": balanced_threshold,
        "threshold_selection_rule": "Maximize validation balanced accuracy; tie-break by higher sensitivity, closer to 0.5, then smaller threshold.",
        "validation_sample_count": int(len(ensemble)),
        "validation_metrics": final_metrics,
        "positive_class": "PNEUMONIA",
        "label_mapping": {"0": "NORMAL", "1": "PNEUMONIA"},
        "test_loaded": False,
        "test_evaluated": False,
        "training_performed": False,
    }
    write_json(output_dir / "final_protocol.json", protocol)
    freeze_protocol(output_dir)
    protocol["final_protocol_sha256"] = sha256_file(output_dir / "final_protocol.json")
    return protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = build_final_protocol(args.registry, args.output_dir)
    print("FINAL_PROTOCOL_DIR=" + str(args.output_dir.resolve()))
    print("SELECTED_CALIBRATION_METHOD=" + protocol["selected_calibration_method"])
    print("BALANCED_THRESHOLD=" + str(protocol["balanced_threshold"]))
    print("FINAL_PROTOCOL_SHA256=" + protocol["final_protocol_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
