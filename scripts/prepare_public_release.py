"""Prepare public-release documentation and assets without training or inference."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
REPORTS = ROOT / "reports"
FINAL_TEST_DIR = ROOT / "results/final_test/evaluation_20260718T090726Z"
PRIMARY_THRESHOLD = 0.5618644666666667

PALETTE = {
    "navy": "#1F3A5F",
    "blue": "#3E6EA8",
    "cyan": "#5BA6A9",
    "amber": "#D99A2B",
    "red": "#B14A4A",
    "green": "#4F8F6F",
    "ink": "#20242A",
    "muted": "#6B7280",
    "grid": "#D8DEE8",
    "paper": "#FFFFFF",
    "panel": "#F6F8FB",
}


def apply_publication_style() -> None:
    """Use a consistent academic plotting style for public figures."""
    plt.rcParams.update({
        "figure.facecolor": PALETTE["paper"],
        "axes.facecolor": PALETTE["paper"],
        "axes.edgecolor": PALETTE["ink"],
        "axes.linewidth": 1.15,
        "axes.labelcolor": PALETTE["ink"],
        "axes.titlecolor": PALETTE["ink"],
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "savefig.bbox": "tight",
        "savefig.facecolor": PALETTE["paper"],
    })


def finish_figure(fig: plt.Figure, path: Path, *, dpi: int = 260) -> None:
    fig.tight_layout(pad=1.2)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def soften_axes(ax: plt.Axes, *, ygrid: bool = True, xgrid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    if ygrid:
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8, alpha=0.85)
    if xgrid:
        ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)


def add_subtitle(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, 0.01, text, fontsize=9, color=PALETTE["muted"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def copy_asset(source: str, target: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / source, ASSETS / target)


def make_pipeline_overview() -> None:
    fig, ax = plt.subplots(figsize=(14, 3.2))
    ax.axis("off")
    steps = [
        ("Data audit", "Filename checks\npatient-id inference"),
        ("v3_clean split", "Patient-level\ntrain / val / test"),
        ("Model families", "CNN baseline\ntransfer learning"),
        ("Multi-seed validation", "Robustness across\n42 / 2025 / 2026"),
        ("Frozen protocol", "Ensemble + calibration\n+ primary threshold"),
        ("One-time test", "Locked final\nevaluation"),
        ("Post-hoc analysis", "Errors + Grad-CAM\nexploratory only"),
    ]
    x = np.linspace(0.07, 0.93, len(steps))
    for i, (xi, (title, body)) in enumerate(zip(x, steps)):
        color = PALETTE["navy"] if i in {4, 5} else PALETTE["blue"]
        ax.text(
            xi, 0.60, f"{title}\n{body}", ha="center", va="center", fontsize=9.3,
            linespacing=1.25, color=PALETTE["ink"],
            bbox=dict(boxstyle="round,pad=0.55,rounding_size=0.12", fc=PALETTE["panel"], ec=color, lw=1.35),
        )
        if i < len(steps) - 1:
            ax.annotate(
                "", xy=(x[i + 1] - 0.06, 0.60), xytext=(xi + 0.06, 0.60),
                arrowprops=dict(arrowstyle="-|>", lw=1.25, color=PALETTE["muted"], shrinkA=0, shrinkB=0),
            )
    ax.text(0.07, 0.91, "Leakage-aware chest X-ray pneumonia classification workflow",
            ha="left", va="center", fontsize=16, fontweight="bold", color=PALETTE["ink"])
    ax.text(0.07, 0.18, "Final-test choices are frozen before test evaluation; Grad-CAM is post-hoc exploratory.",
            ha="left", va="center", fontsize=10, color=PALETTE["muted"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0.08, 0.98)
    finish_figure(fig, ASSETS / "pipeline_overview.png", dpi=260)


def make_class_distribution() -> None:
    splits = ["Train", "Validation", "Test"]
    normal = np.array([1079, 270, 234])
    pneumonia = np.array([2742, 684, 390])
    x = np.arange(len(splits))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    bars_n = ax.bar(x - width / 2, normal, width, color=PALETTE["cyan"], edgecolor=PALETTE["ink"], linewidth=0.8, label="Normal")
    bars_p = ax.bar(x + width / 2, pneumonia, width, color=PALETTE["red"], edgecolor=PALETTE["ink"], linewidth=0.8, label="Pneumonia")
    ax.set_ylabel("Image count")
    ax.set_title("v3_clean image-level class distribution")
    ax.set_xticks(x, splits)
    ax.legend(loc="upper right")
    soften_axes(ax)
    for bars in [bars_n, bars_p]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 35,
                    f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=9)
    add_subtitle(fig, "Counts are image-level; patient-level grouping is used for leakage-aware splitting.")
    finish_figure(fig, ASSETS / "class_distribution.png")


def make_transfer_comparison() -> None:
    labels = ["VGG16 seed42", "EffNetB0 seed42", "EffNetB0 3-seed mean"]
    acc = [0.961216, 0.970650, 0.972397]
    bal = [(0.950292 + 0.988889) / 2, (0.975146 + 0.959259) / 2, 0.969168]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(x - 0.18, acc, 0.36, label="Accuracy", color=PALETTE["blue"], edgecolor=PALETTE["ink"], linewidth=0.8)
    ax.bar(x + 0.18, bal, 0.36, label="Balanced accuracy", color=PALETTE["amber"], edgecolor=PALETTE["ink"], linewidth=0.8)
    ax.set_ylim(0.93, 1.0)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_title("Validation transfer-learning comparison")
    ax.set_ylabel("Validation metric")
    ax.legend(loc="upper left")
    soften_axes(ax)
    add_subtitle(fig, "Validation-only comparison; final protocol selected the EfficientNetB0 control ensemble before test evaluation.")
    finish_figure(fig, ASSETS / "transfer_model_comparison.png")


def make_cnn_multiseed_metrics() -> None:
    metrics = pd.read_csv(ROOT / "results/experiments/cnn_baseline_v1/multiseed_summary/metrics_by_seed.csv")
    metric_cols = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "roc_auc", "pr_auc"]
    labels = ["Accuracy", "Balanced acc.", "Sensitivity", "Specificity", "ROC-AUC", "PR-AUC"]
    x = np.arange(len(metric_cols))
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    for _, row in metrics.iterrows():
        ax.plot(x, row[metric_cols].to_numpy(), marker="o", linewidth=1.35, alpha=0.55, label=f"Seed {int(row['seed'])}")
    mean = metrics[metric_cols].mean(axis=0).to_numpy()
    ax.plot(x, mean, marker="D", markersize=6, linewidth=2.4, color=PALETTE["ink"], label="Mean")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylim(0.50, 1.02)
    ax.set_ylabel("Validation metric")
    ax.set_title("CNN baseline: multi-seed validation profile")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    soften_axes(ax)
    add_subtitle(fig, "Thin lines show individual seeds; the dark line shows the three-seed mean.")
    finish_figure(fig, ASSETS / "cnn_multiseed_metrics.png")


def make_imbalance_strategy_comparison() -> None:
    summary = pd.read_csv(ROOT / "results/experiments/efficientnetb0_imbalance_comparison/strategy_metrics_summary.csv")
    strategy_labels = {"control": "Control", "class_weight": "Class weight", "focal": "Focal loss"}
    summary["label"] = summary["strategy"].map(strategy_labels)
    metrics = [
        ("balanced_accuracy", "Balanced accuracy", PALETTE["blue"]),
        ("sensitivity", "Sensitivity", PALETTE["green"]),
        ("specificity", "Specificity", PALETTE["amber"]),
        ("brier_score", "Brier score", PALETTE["red"]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.8))
    for ax, (prefix, title, color) in zip(axes, metrics):
        means = summary[f"{prefix}_mean"].to_numpy()
        std = summary[f"{prefix}_sample_std"].to_numpy()
        y = np.arange(len(summary))
        ax.barh(y, means, xerr=std, color=color, alpha=0.88, edgecolor=PALETTE["ink"], linewidth=0.8,
                error_kw=dict(ecolor=PALETTE["ink"], lw=1.1, capsize=3))
        ax.set_yticks(y, summary["label"] if ax is axes[0] else [])
        ax.set_title(title)
        ax.set_xlim(0, max(1.0, float(np.nanmax(means + std)) * 1.05) if prefix != "brier_score" else 0.045)
        soften_axes(ax, ygrid=False, xgrid=True)
        if prefix == "brier_score":
            ax.set_xlabel("Lower is better")
        else:
            ax.set_xlabel("Higher is better")
    fig.suptitle("EfficientNetB0 imbalance strategies: validation-only selection evidence", y=1.03, fontsize=15, fontweight="bold")
    add_subtitle(fig, "Bars show three-seed means; whiskers show sample standard deviation.")
    finish_figure(fig, ASSETS / "imbalance_strategy_comparison.png")


def make_final_test_curves() -> None:
    predictions = pd.read_csv(FINAL_TEST_DIR / "final_test_predictions.csv")
    y_true = predictions["true_label"].to_numpy()
    probabilities = predictions["final_probability"].to_numpy()
    predicted = (probabilities >= PRIMARY_THRESHOLD).astype(int)
    tn = int(((y_true == 0) & (predicted == 0)).sum())
    fp = int(((y_true == 0) & (predicted == 1)).sum())
    fn = int(((y_true == 1) & (predicted == 0)).sum())
    tp = int(((y_true == 1) & (predicted == 1)).sum())
    if (tn, fp, fn, tp) != (152, 82, 1, 389):
        raise RuntimeError(f"Unexpected final-test confusion counts: {(tn, fp, fn, tp)}")

    matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
    ax.set_xticks([0, 1], ["Predicted normal", "Predicted pneumonia"])
    ax.set_yticks([0, 1], ["True normal", "True pneumonia"])
    ax.set_title("Final test confusion matrix at frozen threshold")
    for i in range(2):
        for j in range(2):
            value = matrix[i, j]
            color = "white" if value > matrix.max() * 0.55 else PALETTE["ink"]
            ax.text(j, i, f"{value:,}", ha="center", va="center", fontsize=18, fontweight="bold", color=color)
    ax.text(0.5, -0.16, f"Primary threshold = {PRIMARY_THRESHOLD:.4f}; no threshold re-selection",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color=PALETTE["muted"])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Images")
    finish_figure(fig, ASSETS / "final_test_confusion_matrix.png")

    fpr, tpr, _ = roc_curve(y_true, probabilities)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.plot(fpr, tpr, color=PALETTE["blue"], lw=2.6, label=f"Ensemble ROC-AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color=PALETTE["muted"], lw=1.1, ls="--", label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Final test ROC curve")
    ax.legend(loc="lower right")
    soften_axes(ax)
    add_subtitle(fig, "Frozen EfficientNetB0 control ensemble; probabilities were not recalibrated after test evaluation.")
    finish_figure(fig, ASSETS / "final_test_roc_curve.png")

    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    pr_auc = auc(recall, precision)
    prevalence = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.plot(recall, precision, color=PALETTE["green"], lw=2.6, label=f"PR-AUC = {pr_auc:.3f}")
    ax.axhline(prevalence, color=PALETTE["muted"], lw=1.1, ls="--", label=f"Prevalence = {prevalence:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Recall / sensitivity")
    ax.set_ylabel("Precision / PPV")
    ax.set_title("Final test precision-recall curve")
    ax.legend(loc="lower left")
    soften_axes(ax)
    add_subtitle(fig, "Curve is descriptive for the frozen final-test predictions; it is not used for model selection.")
    finish_figure(fig, ASSETS / "final_test_pr_curve.png")

    frac_pos, mean_pred = calibration_curve(y_true, probabilities, n_bins=15, strategy="uniform")
    bins = np.linspace(0, 1, 16)
    bin_ids = np.digitize(probabilities, bins[1:-1], right=False)
    counts = np.bincount(bin_ids, minlength=15)
    nonzero_counts = counts[counts > 0]
    sizes = np.clip(nonzero_counts * 4.0, 30, 500)
    ece_payload = json.loads((FINAL_TEST_DIR / "test_metrics_with_ci.json").read_text(encoding="utf-8"))
    ece = ece_payload["point_estimates"]["balanced_threshold"]["ece"]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.plot([0, 1], [0, 1], color=PALETTE["muted"], lw=1.1, ls="--", label="Perfect calibration")
    ax.scatter(mean_pred, frac_pos, s=sizes, color=PALETTE["amber"], edgecolor=PALETTE["ink"], linewidth=0.8, alpha=0.88,
               label="15 uniform bins")
    ax.plot(mean_pred, frac_pos, color=PALETTE["amber"], lw=1.4, alpha=0.75)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed pneumonia fraction")
    ax.set_title(f"Final test calibration curve (ECE={ece:.3f})")
    ax.legend(loc="upper left")
    soften_axes(ax)
    add_subtitle(fig, "Marker size reflects bin count; calibration worsened on the final test set.")
    finish_figure(fig, ASSETS / "final_test_calibration_curve.png")


def make_selected_gradcam_examples() -> None:
    # Fixed non-cherry-picked subset from the deterministic Stage 8A selection:
    # 1 TN, 2 FP including a high-confidence FP, 1 FN, 2 TP.
    entries = [
        ("TN", "01_TN_IM-0019-0001_overlay.png"),
        ("FP", "15_FP_NORMAL2-IM-0171-0001_overlay.png"),
        ("FP", "00_FP_IM-0010-0001_overlay.png"),
        ("FN", "29_FN_person154_bacteria_728_overlay.png"),
        ("TP", "24_TP_person103_bacteria_488_overlay.png"),
        ("TP", "34_TP_person91_bacteria_446_overlay.png"),
    ]
    tile = 320
    header = 40
    images = []
    for label, name in entries:
        image = Image.open(ROOT / "results/posthoc_analysis/gradcam" / name).convert("RGB").resize((tile, tile))
        panel = Image.new("RGB", (tile, tile + header), "white")
        panel.paste(image, (0, header))
        draw = ImageDraw.Draw(panel)
        color = {"TN": "#4F8F6F", "TP": "#1F3A5F", "FP": "#B14A4A", "FN": "#D99A2B"}[label]
        draw.rectangle((0, 0, tile, header), fill=color)
        draw.text((10, 12), f"{label}  |  seed 42 Grad-CAM", fill="white")
        images.append(panel)
    canvas = Image.new("RGB", (3 * tile, 2 * (tile + header) + 42), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), "Deterministic post-hoc Grad-CAM selection (single seed-42 model)", fill=(32, 36, 42))
    for i, image in enumerate(images):
        canvas.paste(image, ((i % 3) * tile, 42 + (i // 3) * (tile + header)))
    canvas.save(ASSETS / "selected_gradcam_examples.png")


def generate_assets() -> None:
    apply_publication_style()
    ASSETS.mkdir(parents=True, exist_ok=True)
    make_pipeline_overview()
    make_class_distribution()
    make_cnn_multiseed_metrics()
    make_transfer_comparison()
    make_imbalance_strategy_comparison()
    make_final_test_curves()
    make_selected_gradcam_examples()


def sanitize_reports() -> None:
    wsl_d = "/mnt/" + "d/"
    win_project = "D:" + r"\\my project\\chest-X-ray"
    win_wsl_base = "D:" + r"\\WSL\\Ubuntu-D"
    home_path = "/home/" + "evelyn"
    local_user = "茶" + "小栀"
    laptop_prefix = "LAPTOP" + "-"
    replacements = [
        (win_project, "<PROJECT_ROOT>"),
        (r"D:/my project/chest-X-ray", "<PROJECT_ROOT>"),
        (win_wsl_base, "<LOCAL_WSL_BASEPATH>"),
        (r"D:/\.\.\.", "<WINDOWS_ABSOLUTE_PATH>"),
        (r"D:/", "<WINDOWS_DRIVE>/"),
        (re.escape(wsl_d + "my project/chest-X-ray"), "<PROJECT_ROOT>"),
        (re.escape(wsl_d + "my project/chest-x-ray"), "<PROJECT_ROOT>"),
        (re.escape(wsl_d) + r"\.\.\.", "<WSL_MOUNT_PATH>"),
        (re.escape(wsl_d), "<WSL_D_MOUNT>/"),
        (re.escape(home_path + "/.venvs/chest-xray"), "<VENV>"),
        (re.escape(home_path), "<HOME>"),
        (local_user, "<LOCAL_USER>"),
        (laptop_prefix + r"[A-Z0-9-]+", "<LOCAL_MACHINE>"),
    ]
    for path in REPORTS.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        for pattern, repl in replacements:
            text = re.sub(pattern, repl, text)
        path.write_text(text, encoding="utf-8")


def docs_texts() -> dict[str, str]:
    today = date.today().isoformat()
    return {
        "DATASET.md": """
# Dataset

This project uses the Kaggle dataset `paultimothymooney/chest-xray-pneumonia` (Chest X-Ray Images, Pneumonia). Raw images are not distributed in this repository.

The original code and documentation in this repository are released under the MIT License. The chest X-ray data remain governed by the original data provider's terms. ImageNet pretrained weights and third-party dependencies follow their own licenses.

Download example:

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
```

Expected local layout after extraction:

```text
data/raw/chest_xray/
  train/
  val/
  test/
```

The original public dataset has a very small official validation split (16 images), which is not suitable for robust model selection. This project therefore performs a leakage-aware re-splitting process. Patient IDs are inferred from filename patterns because external patient metadata are not provided.

The v3_clean protocol excludes 457 PNEUMONIA images from model development because their inferred patient IDs overlap the official test patients. The final v3_clean counts are:

| split | images |
| --- | ---: |
| train | 3821 |
| validation | 954 |
| test | 624 |

The final test set is the official test split and is isolated until model selection and threshold freezing are complete.

Data licensing and citation should follow the Kaggle dataset page and original data providers. This repository does not re-license the images.
""",
        "EXPERIMENT_PROTOCOL.md": """
# Experiment protocol

This project follows a leakage-aware protocol:

1. Audit original filenames and inferred patient IDs.
2. Remove development images whose inferred patient IDs overlap the official test set.
3. Create a v3_clean patient-level train/validation split.
4. Verify the TensorFlow data pipeline.
5. Train a CNN baseline.
6. Train VGG16 and EfficientNetB0 transfer-learning models.
7. Run multi-seed validation for the selected EfficientNetB0 control strategy.
8. Compare class weighting and focal loss as imbalance strategies.
9. Freeze the final ensemble, calibration method, and thresholds before final test.
10. Run the final test evaluation once.
11. Perform post-hoc exploratory error analysis and Grad-CAM.

The test set was not used for model selection, calibration selection, threshold selection, or imbalance-strategy selection. After the final test evaluation, the model, threshold, and calibration choices are frozen and must not be changed based on test results.
""",
        "REPRODUCIBILITY.md": """
# Reproducibility

The original project was developed on Windows with WSL2 and Python 3.11. The local WSL distribution name is not required for other users.

Core environment:

- Python 3.11
- TensorFlow 2.21
- pandas, numpy, Pillow, PyYAML, matplotlib, pytest, scikit-learn
- NVIDIA GPU optional for training; CPU is sufficient for audit and unit tests but training will be slow.

Typical setup:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install "tensorflow[and-cuda]==2.21.*"
python -m pip install -r requirements.txt
```

Example workflow:

```bash
python src/create_splits_v3_clean.py --help
python src/check_data_pipeline.py --help
python -m pytest -q
python src/train_cnn.py --help
python src/train_transfer.py --help
```

The one-time final-test lock files are part of the original study audit trail and are not required for ordinary public-code reuse. Do not bypass one-time final-test protections when reproducing the protocol.
""",
        "RESULTS.md": """
# Results

All results are research results from a single public dataset. They are not clinical-performance claims.

## Validation: CNN baseline, three seeds

| metric | mean ± sample std |
| --- | ---: |
| accuracy | 0.8924 ± 0.0198 |
| ROC-AUC | 0.9743 ± 0.0063 |
| PR-AUC | 0.9897 ± 0.0029 |

## Validation: transfer learning

| model | seed(s) | accuracy | sensitivity | specificity | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| VGG16 | 42 | 0.9612 | 0.9503 | 0.9889 | 0.9959 |
| EfficientNetB0 control | 42/2025/2026 | 0.972397 ± 0.001601 | 0.976608 ± 0.001462 | 0.961728 ± 0.004277 | 0.996569 ± 0.000433 |

EfficientNetB0 control also achieved validation balanced accuracy 0.969168 ± 0.002260, PR-AUC 0.998689 ± 0.000170, and Brier score 0.020623 ± 0.001454.

## Validation: imbalance strategies

The control strategy was selected. Class weighting did not improve mean balanced accuracy. Focal loss increased specificity but reduced sensitivity, F1, and calibration quality.

## Frozen validation ensemble

At threshold 0.5, the three-seed EfficientNetB0 control ensemble achieved validation accuracy 0.9738, balanced accuracy 0.9716, and ROC-AUC 0.9967. The selected calibration method was none. The primary balanced threshold was fixed at 0.5618644667.

## Frozen official test

| operating point | accuracy | sensitivity | specificity | balanced accuracy | ROC-AUC | PR-AUC | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| benchmark threshold 0.5 | 0.8574 | 0.9974 | 0.6239 | 0.8107 | 0.9751 | 0.9829 | 0.1197 | 0.1438 |
| primary balanced threshold 0.5618644667 | 0.8670 | 0.9974 | 0.6496 | 0.8235 | 0.9751 | 0.9829 | 0.1197 | 0.1438 |

Patient-cluster bootstrap 95% intervals are reported in `reports/final_test_results.md`; they are copied from the frozen final-test report and are not recomputed here.

## Validation-test gap

The final test set preserved high sensitivity but showed substantially lower specificity and worse calibration than validation. This gap is emphasized as a key limitation.

## Post-hoc exploratory analysis

At the primary threshold: TN/FP/FN/TP = 152/82/1/389. There were 53 high-confidence false positives and 13 samples with three-seed label disagreement. The unique false negative was `person154_bacteria_728.jpeg` with final probability 0.1944. Grad-CAM is exploratory and explains only the seed 42 single model, not the ensemble.
""",
        "MODEL_CARD.md": """
# Model card

## Model details

- Model name: EfficientNetB0 control three-seed ensemble
- Version: 0.1.0
- Architecture: EfficientNetB0 transfer learning with a binary classification head
- Ensemble: equal probability average across seeds 42, 2025, and 2026
- Calibration: none
- Thresholds: benchmark 0.5; primary balanced threshold 0.5618644667

## Intended use

Research and education about leakage-aware medical-imaging model evaluation.

## Out-of-scope use

This model must not be used for patient diagnosis, triage, clinical decision-making, or as a substitute for clinicians.

## Data

Training and validation use the v3_clean development split derived from the Kaggle chest X-ray pneumonia dataset. Final evaluation uses the official test split. Patient IDs are inferred from filenames.

## Evaluation

The final official test at the primary threshold achieved accuracy 0.8670, sensitivity 0.9974, specificity 0.6496, balanced accuracy 0.8235, ROC-AUC 0.9751, PR-AUC 0.9829, and Brier score 0.1197.

## Known failure modes

The final test showed many false positives, including 53 high-confidence false positives. Calibration degraded on test relative to validation. No external validation, subgroup analysis, fairness analysis, or device analysis is included.

## Explainability

Grad-CAM visualizations are post-hoc exploratory explanations for the seed 42 model only. They do not prove medical causality or lesion localization correctness.

## Weights

Model weights are not distributed in this repository.

## License and clinical safety

The original code and documentation in this repository use the MIT License. Dataset images, ImageNet pretrained weights, and third-party dependencies remain under their respective licenses. The MIT License is not a clinical-use permission and does not imply medical-device clearance or certification.
""",
        "LIMITATIONS.md": """
# Limitations

1. Single public pediatric chest X-ray dataset.
2. No external validation.
3. Patient IDs are inferred from filenames rather than external metadata.
4. Limited information about acquisition devices, sites, and patient subgroups.
5. Validation and test distributions differ substantially.
6. Test specificity is much lower than validation specificity.
7. Probability calibration worsens on test.
8. High-confidence false positives are common.
9. No subgroup, fairness, or device-level analysis.
10. Grad-CAM does not prove lesion localization correctness.
11. Only three random seeds were used for the main multi-seed experiments.
12. The model is not suitable for clinical deployment.
""",
        "ETHICS_AND_SAFETY.md": """
# Ethics and safety

This repository is for research and education only. It must not be used for patient diagnosis, triage, or as a replacement for clinician decision-making.

The model can produce high-confidence errors. In the final test set, normal images produced many false positives, and calibration worsened compared with validation.

The repository does not distribute chest X-ray images, trained model weights, complete prediction files, or bootstrap detail files. Users must follow the dataset provider's terms and applicable laws.

The original code and documentation are released under the MIT License. The license does not grant clinical-use authorization, medical-device certification, or rights to redistribute the dataset or third-party pretrained weights.
""",
    }


def write_docs() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    for name, text in docs_texts().items():
        write(DOCS / name, text)
    write(ROOT / "README.md", readme_text())
    write(ROOT / "CITATION.cff", citation_text())
    write(ROOT / "configs/examples/data.example.yaml", """
dataset_root: data/raw/chest_xray
train_manifest: data/splits/v3_clean/train.csv
val_manifest: data/splits/v3_clean/val.csv
test_manifest: data/splits/v3_clean/test.csv
image_size: [224, 224]
batch_size: 16
positive_class: PNEUMONIA
split_protocol: v3_clean_patient_level
""")
    write(ROOT / "configs/examples/cnn_baseline.example.yaml", """
experiment_name: cnn_baseline_v1
data_config: configs/data_v3_clean.yaml
seed: 42
image_size: [224, 224]
batch_size: 32
threshold: 0.5
mixed_precision: false
""")
    write(ROOT / "configs/examples/efficientnetb0.example.yaml", """
experiment_name: efficientnetb0_transfer_v1
model_name: efficientnetb0
data_config: configs/data_v3_clean.yaml
seed: 42
image_size: [224, 224]
batch_size: 16
loss: binary_crossentropy
class_weight: null
threshold: 0.5
weights: imagenet
mixed_precision: false
""")


def readme_text() -> str:
    return """
# chest-xray-pneumonia-classification

![Status](https://img.shields.io/badge/status-research%20release-blue)
![Clinical use](https://img.shields.io/badge/clinical%20use-not%20intended-red)
![Python](https://img.shields.io/badge/python-3.11-green)

A reproducible and leakage-aware deep learning study for pediatric chest X-ray pneumonia classification, including patient-level splitting, multi-seed evaluation, frozen test protocol, calibration analysis, and post-hoc explainability.

This is a research and education project. It is not a medical device and must not be used for clinical diagnosis or triage.

## Core question

How much of a strong validation result survives a leakage-aware protocol, multi-seed validation, frozen model selection, and one-time final test evaluation?

## Key result snapshot

The headline is not simply validation accuracy. The final test set showed a meaningful generalization gap, especially lower specificity and worse calibration.

| stage | operating point | accuracy | sensitivity | specificity | balanced accuracy | ROC-AUC | PR-AUC | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | EfficientNetB0 control ensemble, threshold 0.5 | 0.9738 | 0.9766 | 0.9667 | 0.9716 | 0.9967 | 0.9988 | 0.0201 |
| frozen official test | primary threshold 0.5618644667 | 0.8670 | 0.9974 | 0.6496 | 0.8235 | 0.9751 | 0.9829 | 0.1197 |

## Workflow

![Pipeline overview](docs/assets/pipeline_overview.png)

## Data

The project uses the Kaggle dataset `paultimothymooney/chest-xray-pneumonia`. Raw images are not included in this repository. Patient IDs are inferred from filenames, which is a limitation.

See [docs/DATASET.md](docs/DATASET.md).

## Leakage control

The official validation split has only 16 images, so the project builds a v3_clean development split. Development images with inferred patient IDs overlapping the official test set are excluded before train/validation splitting.

## Experiments

- CNN baseline, three seeds
- VGG16 transfer learning
- EfficientNetB0 transfer learning
- EfficientNetB0 multi-seed validation
- Class weighting and focal loss ablations
- Frozen ensemble, calibration, and threshold selection
- One-time final test
- Post-hoc error analysis and Grad-CAM

![CNN multi-seed metrics](docs/assets/cnn_multiseed_metrics.png)
![Imbalance strategy comparison](docs/assets/imbalance_strategy_comparison.png)

## Final test and errors

At the primary frozen threshold, test confusion counts were TN=152, FP=82, FN=1, TP=389. The main error mode was false positives; 53 FP cases were high-confidence by the fixed post-hoc definition.

![Final test confusion matrix](docs/assets/final_test_confusion_matrix.png)
![Final test ROC](docs/assets/final_test_roc_curve.png)
![Final test PR](docs/assets/final_test_pr_curve.png)
![Final test calibration](docs/assets/final_test_calibration_curve.png)

## Explainability

Grad-CAM examples are post-hoc exploratory visualizations from the seed 42 model only. They do not explain the full ensemble and do not prove medical causality.

![Selected Grad-CAM examples](docs/assets/selected_gradcam_examples.png)

## Project structure

```text
configs/     experiment and data configuration
src/         data audit, splitting, training, aggregation, protocol, and analysis code
scripts/     environment, run, and validation scripts
tests/       synthetic and unit tests
docs/        public documentation
reports/     public Markdown reports
```

## Environment

Use Python 3.11. TensorFlow 2.21 was used in the original WSL2 environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Data preparation

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
python src/create_splits_v3_clean.py --help
python src/check_data_pipeline.py --help
```

Real manifests are generated locally and are not committed.

## Reproducibility commands

```bash
python -m pytest -q
python src/train_cnn.py --help
python src/train_transfer.py --help
python src/aggregate_transfer_multiseed.py --help
```

The original final test was protected by a one-time evaluation protocol. Do not use test results to change thresholds, calibration, or model choice.

## Documentation

- [Dataset](docs/DATASET.md)
- [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md)
- [Results](docs/RESULTS.md)
- [Model card](docs/MODEL_CARD.md)
- [Limitations](docs/LIMITATIONS.md)
- [Ethics and safety](docs/ETHICS_AND_SAFETY.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)

## Citation

See [CITATION.cff](CITATION.cff).

## License

The original code and documentation in this repository are released under the MIT License. Chest X-ray images are not distributed here and remain governed by the data provider's terms. ImageNet pretrained weights and third-party dependencies follow their respective licenses. Model weights are not distributed. The MIT License does not authorize clinical use or imply medical-device certification.

## 中文概述

本项目是一个儿童胸片肺炎分类的研究/教学项目，重点在于患者级划分、防止数据泄漏、多随机种子验证、冻结协议和一次性最终测试。模型不能用于临床诊断或分诊。
"""


def citation_text() -> str:
    return """
cff-version: 1.2.0
message: "If you use this repository, please cite it using this metadata."
title: "chest-xray-pneumonia-classification"
authors:
  - family-names: "Zhao"
    given-names: "Xiaorui"
version: "0.1.0"
repository-code: "https://github.com/Evelyn-R7/chest-xray-pneumonia-classification"
url: "https://github.com/Evelyn-R7/chest-xray-pneumonia-classification"
license: "MIT"
"""


def update_gitignore() -> None:
    required = [
        "data/raw/", "data/processed/", "data/splits/", "results/", "*.keras", "*.h5",
        "*.joblib", "*.zip", "*.csv", "*.npy", "*.npz", "*.log", ".venv/",
        ".venv-wsl/", "__pycache__/", "**/__pycache__/", ".pytest_cache/",
        ".ipynb_checkpoints/", "reports/data_pipeline/", "reports/data_audit.py",
    ]
    path = ROOT / ".gitignore"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = existing.splitlines()
    for item in required:
        if item not in lines:
            lines.append(item)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def public_manifest() -> None:
    public_paths = []
    for pattern in ["README.md", "LICENSE", "CITATION.cff", "requirements.txt", ".gitignore", "configs/**/*.yaml", "src/**/*.py", "scripts/**/*.sh", "scripts/**/*.py", "tests/**/*.py", "docs/**/*", "reports/*.md"]:
        public_paths.extend(ROOT.glob(pattern))
    records = []
    for path in sorted(set(public_paths)):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel == "docs/PUBLIC_RESULTS_MANIFEST.json":
            continue
        category = "documentation"
        if "final_test" in rel:
            category = "test"
        elif "posthoc" in rel or "gradcam" in rel:
            category = "post-hoc"
        elif "validation" in rel or "multiseed" in rel or "transfer" in rel or "imbalance" in rel:
            category = "validation"
        records.append({
            "relative_path": rel,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "source": "public release file generated or curated from existing reports/results",
            "category": category,
            "contains_image": path.suffix.lower() in {".png", ".jpg", ".jpeg"},
            "suitable_for_public_release": True,
        })
    write(DOCS / "PUBLIC_RESULTS_MANIFEST.json", json.dumps(records, indent=2))


def main() -> None:
    generate_assets()
    sanitize_reports()
    write_docs()
    update_gitignore()
    public_manifest()


if __name__ == "__main__":
    main()
