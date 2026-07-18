"""Post-hoc exploratory error analysis and Grad-CAM for the frozen final test.

This module is deliberately post-hoc: it must not train, re-select thresholds,
modify the frozen protocol, or re-run full test inference.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from src.final_test.inference import resolve_manifest_image_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_TEST_DIR = PROJECT_ROOT / "results/final_test/evaluation_20260718T090726Z"
PREDICTIONS_PATH = FINAL_TEST_DIR / "final_test_predictions.csv"
PATIENT_PREDICTIONS_PATH = FINAL_TEST_DIR / "patient_level_test_predictions.csv"
TEST_MANIFEST_PATH = PROJECT_ROOT / "data/splits/v3_clean/test.csv"
FINAL_PROTOCOL_PATH = PROJECT_ROOT / "results/final_protocol/final_protocol.json"
OUTPUT_DIR = PROJECT_ROOT / "results/posthoc_analysis"
REPORT_PATH = PROJECT_ROOT / "reports/posthoc_error_analysis.md"
PRIMARY_THRESHOLD = 0.5618644666666667
EXPECTED_COUNTS = {"TN": 152, "FP": 82, "FN": 1, "TP": 389}
SEEDS = (42, 2025, 2026)


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]

    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.12g}"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt(row[column]) for column in frame.columns) + " |")
    return "\n".join(lines)


def ensure_primary_threshold(threshold: float) -> float:
    if float(threshold) != PRIMARY_THRESHOLD:
        raise ValueError(f"Only frozen primary threshold {PRIMARY_THRESHOLD} is allowed")
    return PRIMARY_THRESHOLD


def add_error_groups(predictions: pd.DataFrame, threshold: float = PRIMARY_THRESHOLD) -> pd.DataFrame:
    ensure_primary_threshold(threshold)
    required = {
        "filename", "patient_id", "true_label", "final_probability", "std_probability",
        "probability_seed_42", "probability_seed_2025", "probability_seed_2026",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    output = predictions.copy()
    output["predicted_primary"] = (output["final_probability"].astype(float) >= threshold).astype(int)
    conditions = [
        (output["true_label"].astype(int) == 0) & (output["predicted_primary"] == 0),
        (output["true_label"].astype(int) == 0) & (output["predicted_primary"] == 1),
        (output["true_label"].astype(int) == 1) & (output["predicted_primary"] == 0),
        (output["true_label"].astype(int) == 1) & (output["predicted_primary"] == 1),
    ]
    output["error_group"] = np.select(conditions, ["TN", "FP", "FN", "TP"], default="INVALID")
    counts = output["error_group"].value_counts().to_dict()
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected primary-threshold confusion counts: {counts}")
    seed_labels = np.column_stack([
        (output[f"probability_seed_{seed}"].astype(float).to_numpy() >= threshold).astype(int)
        for seed in SEEDS
    ])
    output["seed_label_disagreement"] = seed_labels.max(axis=1) != seed_labels.min(axis=1)
    output["distance_to_threshold"] = (output["final_probability"].astype(float) - threshold).abs()
    output["high_confidence_error"] = (
        ((output["error_group"] == "FP") & (output["final_probability"].astype(float) >= 0.90))
        | ((output["error_group"] == "FN") & (output["final_probability"].astype(float) <= 0.10))
    )
    output["near_threshold"] = output["distance_to_threshold"] <= 0.05
    return output


def write_error_cases(grouped: pd.DataFrame, output_dir: Path) -> Path:
    columns = [
        "filename", "patient_id", "true_label", "final_probability", "std_probability",
        "error_group", "distance_to_threshold", "probability_seed_42",
        "probability_seed_2025", "probability_seed_2026", "seed_label_disagreement",
    ]
    frame = grouped[columns].copy()
    if any("path" in column.lower() for column in frame.columns):
        raise ValueError("error cases output must not contain paths")
    path = output_dir / "test_error_cases.csv"
    frame.to_csv(path, index=False)
    return path


def grouped_summary(grouped: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, frame in grouped.groupby("error_group", sort=False):
        probs = frame["final_probability"].astype(float)
        stds = frame["std_probability"].astype(float)
        rows.append({
            "error_group": group,
            "count": int(len(frame)),
            "patient_count": int(frame["patient_id"].nunique()),
            "probability_mean": float(probs.mean()),
            "probability_median": float(probs.median()),
            "probability_q1": float(probs.quantile(0.25)),
            "probability_q3": float(probs.quantile(0.75)),
            "std_probability_mean": float(stds.mean()),
            "std_probability_median": float(stds.median()),
            "seed_label_disagreement_count": int(frame["seed_label_disagreement"].sum()),
            "high_confidence_error_count": int(frame["high_confidence_error"].sum()),
            "near_threshold_count": int(frame["near_threshold"].sum()),
        })
    return pd.DataFrame(rows).sort_values("error_group")


def patient_error_analysis(grouped: pd.DataFrame, patient_predictions: pd.DataFrame) -> dict[str, Any]:
    frame = grouped.copy()
    frame["is_error"] = frame["error_group"].isin(["FP", "FN"])
    by_patient = frame.groupby("patient_id", sort=False).agg(
        true_label=("true_label", "first"),
        image_count=("filename", "size"),
        error_count=("is_error", "sum"),
        probability_std=("final_probability", "std"),
        fp_count=("error_group", lambda s: int((s == "FP").sum())),
    ).reset_index()
    by_patient["probability_std"] = by_patient["probability_std"].fillna(0.0)
    by_patient["has_mixed_correctness"] = (by_patient["error_count"] > 0) & (by_patient["error_count"] < by_patient["image_count"])
    single = by_patient[by_patient["image_count"] == 1]
    multi = by_patient[by_patient["image_count"] > 1]
    fp_patients = by_patient[by_patient["fp_count"] > 0].sort_values(["fp_count", "image_count"], ascending=False)
    return {
        "patient_count": int(len(by_patient)),
        "single_image_patient_count": int(len(single)),
        "multi_image_patient_count": int(len(multi)),
        "single_image_error_rate": float(single["error_count"].sum() / single["image_count"].sum()) if len(single) else 0.0,
        "multi_image_error_rate": float(multi["error_count"].sum() / multi["image_count"].sum()) if len(multi) else 0.0,
        "mixed_correctness_patient_count": int(by_patient["has_mixed_correctness"].sum()),
        "mean_within_patient_probability_std": float(by_patient["probability_std"].mean()),
        "fp_patient_count": int(len(fp_patients)),
        "top_fp_patients": fp_patients.head(10).to_dict(orient="records"),
    }


def select_gradcam_samples(grouped: pd.DataFrame, threshold: float = PRIMARY_THRESHOLD) -> pd.DataFrame:
    ensure_primary_threshold(threshold)
    reasons: dict[int, list[str]] = defaultdict(list)

    def add(frame: pd.DataFrame, reason: str) -> None:
        for idx in frame.index:
            reasons[int(idx)].append(reason)

    add(grouped[grouped["error_group"] == "FN"], "all_fn")
    fp = grouped[grouped["error_group"] == "FP"]
    add(fp.sort_values("final_probability", ascending=False).head(5), "fp_top_probability")
    add(fp.sort_values("distance_to_threshold", ascending=True).head(5), "fp_near_threshold")
    add(fp.sort_values("std_probability", ascending=False).head(5), "fp_high_seed_std")
    tn = grouped[grouped["error_group"] == "TN"]
    add(tn.sort_values("final_probability", ascending=True).head(5), "tn_lowest_probability")
    add(tn.sort_values("distance_to_threshold", ascending=True).head(5), "tn_near_threshold")
    tp = grouped[grouped["error_group"] == "TP"]
    add(tp.sort_values("final_probability", ascending=False).head(5), "tp_top_probability")
    add(tp.sort_values("distance_to_threshold", ascending=True).head(5), "tp_near_threshold")

    selected = grouped.loc[sorted(reasons)].copy()
    selected["selection_reason"] = [";".join(reasons[int(idx)]) for idx in selected.index]
    return selected.reset_index(drop=True)


def merge_manifest_paths(selection: pd.DataFrame, manifest_path: Path, project_root: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    subset = manifest[["filename", "relative_path", "filepath"]].copy()
    merged = selection.merge(subset, on="filename", how="left", validate="one_to_one")
    if merged["relative_path"].isna().any():
        raise ValueError("Grad-CAM selection contains filenames missing from manifest")
    merged["resolved_image_path"] = [
        str(resolve_manifest_image_path(value, project_root))
        for value in merged["filepath"]
    ]
    return merged


def find_last_conv_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    """Find the last top-level feature layer connected to model inputs.

    EfficientNetB0 is nested as one Functional layer in this project. Its output
    is the final 7x7 convolutional feature map, so this function prefers the
    last reachable rank-4 layer in the top-level graph instead of hard-coding an
    internal layer name that may not be symbolically connected to model.inputs.
    """

    candidates: list[tf.keras.layers.Layer] = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        try:
            shape = layer.output.shape
        except Exception:
            continue
        if len(shape) != 4:
            continue
        candidates.append(layer)
    if not candidates:
        raise ValueError("No suitable connected rank-4 convolutional feature layer found for Grad-CAM")
    return candidates[-1]


def load_image_array(path: str | Path, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(size)
    return np.asarray(image, dtype=np.float32)


def gradcam_heatmap(model: tf.keras.Model, image: np.ndarray, conv_layer: tf.keras.layers.Layer) -> np.ndarray:
    sample = tf.expand_dims(image, axis=0)
    _ = model(sample, training=False)
    try:
        layer_index = model.layers.index(conv_layer)
    except ValueError as exc:
        raise ValueError("Grad-CAM feature layer must be a top-level connected model layer") from exc
    with tf.GradientTape() as tape:
        conv_outputs = conv_layer(sample, training=False)
        tape.watch(conv_outputs)
        x = conv_outputs
        for layer in model.layers[layer_index + 1:]:
            try:
                x = layer(x, training=False)
            except TypeError:
                x = layer(x)
        predictions = x
        target = predictions[:, 0]
    grads = tape.gradient(target, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv_outputs[0] * weights[0], axis=-1)
    cam = tf.maximum(cam, 0)
    denom = tf.reduce_max(cam)
    if float(denom) <= 0.0:
        return np.zeros(cam.shape, dtype=np.float32)
    return (cam / denom).numpy().astype(np.float32)


def save_gradcam_triplet(
    image: np.ndarray,
    heatmap: np.ndarray,
    output_prefix: Path,
    title: str,
) -> tuple[Path, Path, Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    heat_resized = np.asarray(Image.fromarray(np.uint8(255 * heatmap)).resize((image.shape[1], image.shape[0])), dtype=float) / 255.0
    cmap = plt.get_cmap("jet")(heat_resized)[..., :3]
    overlay = np.clip(0.55 * (image / 255.0) + 0.45 * cmap, 0, 1)
    original_path = output_prefix.with_name(output_prefix.name + "_original.png")
    heatmap_path = output_prefix.with_name(output_prefix.name + "_heatmap.png")
    overlay_path = output_prefix.with_name(output_prefix.name + "_overlay.png")
    Image.fromarray(np.uint8(image)).save(original_path)
    plt.imsave(heatmap_path, heat_resized, cmap="jet")
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.imshow(overlay)
    axis.axis("off")
    axis.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(overlay_path, dpi=160)
    plt.close(fig)
    return original_path, heatmap_path, overlay_path


def run_gradcam(selection: pd.DataFrame, protocol: dict[str, Any], output_dir: Path) -> pd.DataFrame:
    model_path = protocol["best_model_paths"]["seed_42"]
    model = tf.keras.models.load_model(model_path, compile=False)
    try:
        conv_layer = find_last_conv_layer(model)
        rows = []
        gradcam_dir = output_dir / "gradcam"
        for idx, row in selection.iterrows():
            image = load_image_array(row["resolved_image_path"])
            heatmap = gradcam_heatmap(model, image, conv_layer)
            safe_stem = f"{idx:02d}_{row['error_group']}_{Path(row['filename']).stem}"
            title = (
                f"{row['filename']} | true={int(row['true_label'])} | {row['error_group']} | "
                f"ensemble={float(row['final_probability']):.3f} | seed42={float(row['probability_seed_42']):.3f} | "
                f"{row['selection_reason']}"
            )
            original, heat, overlay = save_gradcam_triplet(image, heatmap, gradcam_dir / safe_stem, title)
            rows.append({
                "filename": row["filename"],
                "patient_id": row["patient_id"],
                "true_label": int(row["true_label"]),
                "error_group": row["error_group"],
                "frozen_ensemble_probability": float(row["final_probability"]),
                "seed_42_probability": float(row["probability_seed_42"]),
                "selection_reason": row["selection_reason"],
                "conv_layer": conv_layer.name,
                "original_image": str(original.relative_to(output_dir)),
                "heatmap": str(heat.relative_to(output_dir)),
                "overlay": str(overlay.relative_to(output_dir)),
            })
        return pd.DataFrame(rows)
    finally:
        del model
        tf.keras.backend.clear_session()


def write_plots(grouped: pd.DataFrame, patient_stats: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 5))
    for group, frame in grouped.groupby("error_group"):
        axis.hist(frame["final_probability"], bins=20, alpha=0.55, label=group)
    axis.axvline(PRIMARY_THRESHOLD, color="black", linestyle="--", label="primary threshold")
    axis.set_xlabel("Final probability")
    axis.set_ylabel("Image count")
    axis.set_title("Post-hoc final test probability distributions")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "error_probability_distributions.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5))
    grouped.boxplot(column="std_probability", by="error_group", ax=axis)
    axis.set_title("Seed probability variability by error group")
    axis.figure.suptitle("")
    axis.set_ylabel("std probability")
    fig.tight_layout()
    fig.savefig(output_dir / "error_seed_variability.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5))
    correct = grouped["error_group"].isin(["TN", "TP"])
    axis.scatter(grouped["final_probability"], grouped["std_probability"], c=correct.map({True: "tab:green", False: "tab:red"}), alpha=0.75)
    axis.axvline(PRIMARY_THRESHOLD, color="black", linestyle="--")
    axis.set_xlabel("Final probability")
    axis.set_ylabel("Seed std probability")
    axis.set_title("Confidence vs correctness (post-hoc)")
    fig.tight_layout()
    fig.savefig(output_dir / "confidence_vs_correctness.png", dpi=160)
    plt.close(fig)

    patient_frame = grouped.assign(is_error=grouped["error_group"].isin(["FP", "FN"])).groupby("patient_id").agg(
        image_count=("filename", "size"), error_rate=("is_error", "mean")
    ).reset_index()
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(patient_frame["image_count"], patient_frame["error_rate"], alpha=0.75)
    axis.set_xlabel("Images per patient")
    axis.set_ylabel("Patient image-level error rate")
    axis.set_title("Patient image count vs error rate")
    fig.tight_layout()
    fig.savefig(output_dir / "patient_image_count_vs_error.png", dpi=160)
    plt.close(fig)


def write_overview(gradcam_index: pd.DataFrame, output_dir: Path) -> None:
    images = []
    for _, row in gradcam_index.head(12).iterrows():
        images.append((output_dir / row["overlay"], row["error_group"], row["selection_reason"]))
    if not images:
        return
    cols = 4
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, (path, group, reason) in zip(axes.flat, images):
        axis.imshow(Image.open(path))
        axis.set_title(f"{group}: {reason[:40]}", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_examples_overview.png", dpi=160)
    plt.close(fig)


def write_reports(
    summary: pd.DataFrame,
    patient_stats: dict[str, Any],
    grouped: pd.DataFrame,
    gradcam_index: pd.DataFrame,
    output_dir: Path,
    report_path: Path,
) -> None:
    fp = grouped[grouped["error_group"] == "FP"]
    fn = grouped[grouped["error_group"] == "FN"]
    disagreement_count = int(grouped["seed_label_disagreement"].sum())
    high_conf_fp = int(((fp["final_probability"] >= 0.90)).sum())
    near_fp = int(fp["near_threshold"].sum())
    fn_desc = fn[["filename", "patient_id", "final_probability", "probability_seed_42", "probability_seed_2025", "probability_seed_2026", "std_probability"]].to_dict(orient="records")
    text = f"""# Post-hoc exploratory final-test error analysis

This analysis is post-hoc exploratory. It was performed after the frozen final test evaluation and must not be used to modify the model, calibration, thresholds, or ensemble strategy.

Primary threshold: `{PRIMARY_THRESHOLD}`. No new threshold was selected.

## Confusion groups

{markdown_table(summary)}

The dominant error type is false positive: `{len(fp)}` FP versus `{len(fn)}` FN. High-confidence FP is fixed as final_probability >= 0.90; count = `{high_conf_fp}`. Near-threshold is fixed as abs(probability - threshold) <= 0.05; FP near-threshold count = `{near_fp}`.

## Unique FN

```json
{json.dumps(fn_desc, indent=2)}
```

## Seed stability

Three-seed label disagreement count: `{disagreement_count}` of `{len(grouped)}` images. FP disagreement count: `{int(fp['seed_label_disagreement'].sum())}`. TN disagreement count: `{int(grouped.loc[grouped['error_group']=='TN', 'seed_label_disagreement'].sum())}`.

These are descriptive associations only and do not establish causality.

## Patient-level exploratory analysis

```json
{json.dumps(patient_stats, indent=2)}
```

FP concentration should be interpreted cautiously because patient_id is inferred from filenames.

## Grad-CAM

Grad-CAM was generated for a deterministic sample list using the frozen seed 42 model only. It explains a single model, not the three-model ensemble. Grad-CAM cannot prove medical causality or confirm lesion localization. Heatmaps must not be used as clinical evidence by themselves.

Grad-CAM outputs generated: `{len(gradcam_index)}` samples.

Selection rules: all FN; FP top 5 probability, FP 5 nearest threshold, FP 5 highest seed std; TN lowest 5 probability and 5 nearest threshold; TP top 5 probability and 5 nearest threshold. Duplicates were kept once with all reasons recorded.

No training, recalibration, threshold adjustment, model selection, or protocol change was performed.

Limitations: single public dataset, no external clinical validation, inferred patient IDs, and post-hoc visual explanations. The model is not clinically deployable and is not a replacement for clinicians.
"""
    (output_dir / "posthoc_error_summary.md").write_text(text, encoding="utf-8")
    report_path.write_text(text, encoding="utf-8")


def run_posthoc_analysis() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    patient_predictions = pd.read_csv(PATIENT_PREDICTIONS_PATH)
    protocol = read_json(FINAL_PROTOCOL_PATH)
    grouped = add_error_groups(predictions, PRIMARY_THRESHOLD)
    write_error_cases(grouped, OUTPUT_DIR)
    summary = grouped_summary(grouped)
    summary.to_csv(OUTPUT_DIR / "error_group_summary.csv", index=False)
    patient_stats = patient_error_analysis(grouped, patient_predictions)
    (OUTPUT_DIR / "patient_error_analysis.json").write_text(json.dumps(patient_stats, indent=2) + "\n", encoding="utf-8")
    selection = select_gradcam_samples(grouped, PRIMARY_THRESHOLD)
    selection_for_csv = selection.drop(columns=[column for column in selection.columns if "path" in column.lower()], errors="ignore")
    selection_for_csv.to_csv(OUTPUT_DIR / "gradcam_selection.csv", index=False)
    selection_with_paths = merge_manifest_paths(selection, TEST_MANIFEST_PATH, PROJECT_ROOT)
    gradcam_index = run_gradcam(selection_with_paths, protocol, OUTPUT_DIR)
    gradcam_index.to_csv(OUTPUT_DIR / "gradcam_index.csv", index=False)
    write_plots(grouped, patient_stats, OUTPUT_DIR)
    write_overview(gradcam_index, OUTPUT_DIR)
    write_reports(summary, patient_stats, grouped, gradcam_index, OUTPUT_DIR, REPORT_PATH)
    result = {
        "fp_high_confidence": int(((grouped["error_group"] == "FP") & (grouped["final_probability"] >= 0.90)).sum()),
        "fp_near_threshold": int(((grouped["error_group"] == "FP") & grouped["near_threshold"]).sum()),
        "fn_records": grouped[grouped["error_group"] == "FN"][[
            "filename", "patient_id", "final_probability", "probability_seed_42", "probability_seed_2025", "probability_seed_2026", "std_probability",
        ]].to_dict(orient="records"),
        "seed_label_disagreement_count": int(grouped["seed_label_disagreement"].sum()),
        "fp_patient_count": patient_stats["fp_patient_count"],
        "gradcam_count": int(len(gradcam_index)),
    }
    (OUTPUT_DIR / "posthoc_analysis_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> int:
    parse_args()
    result = run_posthoc_analysis()
    print("POSTHOC_ANALYSIS_DIR=" + str(OUTPUT_DIR.resolve()))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
