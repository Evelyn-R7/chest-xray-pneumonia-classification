"""Execute the frozen final test evaluation exactly once.

This entry point is intentionally not run by Codex during Stage 7B.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.final_protocol.calibration import calibration_metrics
from src.final_test.bootstrap import cluster_bootstrap, metrics_for_two_thresholds, patient_level_predictions
from src.final_test.inference import (
    add_threshold_predictions,
    build_ensemble_prediction_frame,
    load_and_validate_test_manifest,
    predict_single_model,
)
from src.final_test.reporting import (
    file_hashes,
    save_bootstrap_intervals,
    save_standard_plots,
    write_final_test_report,
    write_json,
)
from src.final_test.safety import (
    BENCHMARK_THRESHOLD,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    EXPECTED_BALANCED_THRESHOLD,
    EXPECTED_PROTOCOL_SHA256,
    check_one_time_markers,
    create_evaluated_marker,
    create_started_marker,
    read_json,
    sha256_file,
    utc_now_iso,
    verify_confirmation,
    verify_frozen_marker,
    verify_model_hashes,
    verify_protocol_fields,
    verify_protocol_hash,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--confirm-one-time-test", required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def build_generalization_gap(protocol: dict, test_metrics: dict) -> dict:
    validation = protocol["validation_metrics"]
    return {
        "threshold_0_5": {
            key: float(test_metrics["threshold_0_5"][key]) - float(validation["benchmark_threshold_metrics"][key])
            for key in ("accuracy", "precision", "sensitivity", "specificity", "f1", "balanced_accuracy")
        },
        "balanced_threshold": {
            key: float(test_metrics["balanced_threshold"][key]) - float(validation["balanced_threshold_metrics"][key])
            for key in ("accuracy", "precision", "sensitivity", "specificity", "f1", "balanced_accuracy")
        },
        "threshold_independent": {
            key: float(test_metrics["threshold_0_5"][key]) - float(validation["benchmark_threshold_metrics"][key])
            for key in ("roc_auc", "pr_auc", "brier_score", "log_loss")
        },
    }


def main() -> int:
    args = parse_args()
    verify_confirmation(args.expected_protocol_sha256, args.confirm_one_time_test)
    root = args.project_root
    final_test_dir = root / "results/final_test"
    check_one_time_markers(final_test_dir)

    protocol_path = root / "results/final_protocol/final_protocol.json"
    marker_path = root / "results/final_protocol/PROTOCOL_FROZEN"
    frozen_protocol_sha = verify_protocol_hash(protocol_path, EXPECTED_PROTOCOL_SHA256)
    marker_sha = verify_frozen_marker(marker_path, EXPECTED_PROTOCOL_SHA256)
    protocol = read_json(protocol_path)
    verify_protocol_fields(protocol)
    model_paths = verify_model_hashes(protocol)

    # STARTED is created immediately before reading test manifest content.
    timestamp = utc_now_iso().replace(":", "").replace("-", "").replace("Z", "Z")
    in_progress = final_test_dir / f"_in_progress_{timestamp}"
    final_dir = final_test_dir / f"evaluation_{timestamp}"
    create_started_marker(final_test_dir, {
        "started_at": utc_now_iso(),
        "in_progress_dir": str(in_progress),
        "final_protocol_sha256": frozen_protocol_sha,
    })
    in_progress.mkdir(parents=True, exist_ok=False)

    test_manifest = root / "data/splits/v3_clean/test.csv"
    manifest_sha = sha256_file(test_manifest)
    frame = load_and_validate_test_manifest(
        test_manifest,
        root,
        protocol["manifest_sha256"]["data/splits/v3_clean/test.csv"],
        root / "data/splits/v3_clean/train.csv",
        root / "data/splits/v3_clean/val.csv",
    )

    import tensorflow as tf
    from src.data_pipeline import build_dataset

    dataset = build_dataset(
        test_manifest,
        root,
        image_size=(224, 224),
        batch_size=16,
        training=False,
        augment=False,
        expected_split="test",
        prefetch=False,
        num_parallel_calls=1,
    )
    seed_frames = {}
    audit = {}
    y_true = frame["label"].map({"NORMAL": 0, "PNEUMONIA": 1}).to_numpy(int)
    for seed in (42, 2025, 2026):
        key = f"seed_{seed}"
        probs = predict_single_model(model_paths[key], dataset, expected_count=len(frame))
        seed_frames[seed] = pd.DataFrame({
            "filename": frame["filename"].to_numpy(),
            "patient_id": frame["patient_id"].to_numpy(),
            "true_label": y_true,
            f"probability_seed_{seed}": probs,
        })
        audit[key] = {"model_path": str(model_paths[key]), "prediction_count": int(len(probs))}
        tf.keras.backend.clear_session()

    predictions = add_threshold_predictions(
        build_ensemble_prediction_frame(seed_frames),
        BENCHMARK_THRESHOLD,
        EXPECTED_BALANCED_THRESHOLD,
    )
    predictions.to_csv(in_progress / "final_test_predictions.csv", index=False)
    patient_predictions = patient_level_predictions(predictions)
    patient_predictions.to_csv(in_progress / "patient_level_test_predictions.csv", index=False)

    probs = predictions["final_probability"].to_numpy(float)
    point = metrics_for_two_thresholds(y_true, probs, BENCHMARK_THRESHOLD, EXPECTED_BALANCED_THRESHOLD)
    write_json(in_progress / "test_metrics_point_estimates.json", point)
    bootstrap_table, bootstrap_summary = cluster_bootstrap(
        predictions,
        BENCHMARK_THRESHOLD,
        EXPECTED_BALANCED_THRESHOLD,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    bootstrap_table.to_csv(in_progress / "bootstrap_metrics.csv", index=False)
    write_json(in_progress / "bootstrap_summary.json", bootstrap_summary)
    with_ci = {"point_estimates": point, "bootstrap_ci": bootstrap_summary}
    write_json(in_progress / "test_metrics_with_ci.json", with_ci)
    write_json(in_progress / "model_inference_audit.json", audit)

    patient_y = patient_predictions["true_label"].to_numpy(int)
    patient_p = patient_predictions["final_probability"].to_numpy(float)
    patient_metrics = metrics_for_two_thresholds(patient_y, patient_p, BENCHMARK_THRESHOLD, EXPECTED_BALANCED_THRESHOLD)
    write_json(in_progress / "environment.json", {"evaluation_time": utc_now_iso(), "pid": os.getpid()})
    resolved = {
        "protocol": protocol,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "calibration_metrics_test": calibration_metrics(y_true, probs),
    }
    write_json(in_progress / "resolved_final_test_protocol.json", resolved)
    gap = build_generalization_gap(protocol, point)
    write_json(in_progress / "generalization_gap.json", gap)
    save_standard_plots(y_true, probs, in_progress, BENCHMARK_THRESHOLD, EXPECTED_BALANCED_THRESHOLD)
    save_bootstrap_intervals(bootstrap_summary, in_progress / "bootstrap_intervals.png")
    write_final_test_report(
        in_progress / "final_test_summary.md",
        {
            "frozen_protocol_sha256": frozen_protocol_sha,
            "test_manifest_sha256": manifest_sha,
            "benchmark_threshold": BENCHMARK_THRESHOLD,
            "balanced_threshold": EXPECTED_BALANCED_THRESHOLD,
        },
        point,
        bootstrap_summary,
        patient_metrics,
        gap,
        [],
    )
    shutil.copy2(in_progress / "final_test_summary.md", root / "reports/final_test_results.md")
    hashes = file_hashes({
        "predictions": in_progress / "final_test_predictions.csv",
        "metrics": in_progress / "test_metrics_point_estimates.json",
        "report": in_progress / "final_test_summary.md",
    })
    record = {
        "evaluation_version": "stage_7b_v1",
        "evaluation_time": utc_now_iso(),
        "frozen_protocol_sha256": frozen_protocol_sha,
        "protocol_frozen_marker_sha256": marker_sha,
        "test_manifest_sha256": manifest_sha,
        "model_paths": {key: str(path) for key, path in model_paths.items()},
        "model_sha256": protocol["best_model_sha256"],
        "test_image_count": int(len(predictions)),
        "test_patient_count": int(predictions["patient_id"].nunique()),
        "ensemble_method": "equal probability average across seeds 42, 2025, 2026",
        "calibration_method": "none",
        "benchmark_threshold": BENCHMARK_THRESHOLD,
        "balanced_threshold": EXPECTED_BALANCED_THRESHOLD,
        "bootstrap_method": "patient-level stratified cluster bootstrap",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "prediction_file_sha256": hashes["predictions"],
        "metrics_file_sha256": hashes["metrics"],
        "report_file_sha256": hashes["report"],
        "training_performed": False,
        "threshold_changed": False,
        "calibration_changed": False,
        "models_changed": False,
        "test_evaluated": True,
    }
    write_json(in_progress / "final_test_evaluation.json", record)
    os.rename(in_progress, final_dir)
    create_evaluated_marker(final_test_dir, {
        "evaluated_at": utc_now_iso(),
        "evaluation_run_directory": str(final_dir),
        "final_protocol_json_sha256": frozen_protocol_sha,
        "test_manifest_sha256": manifest_sha,
        "test_predictions_sha256": hashes["predictions"],
        "test_metrics_sha256": hashes["metrics"],
        "no_reselection_statement": "Do not retrain, recalibrate, reweight, or change thresholds based on final test results.",
    })
    print("FINAL_TEST_EVALUATION_DIR=" + str(final_dir.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
