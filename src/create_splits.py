from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

SEED = 42
VAL_RATIO = 0.20
LABELS = ("NORMAL", "PNEUMONIA")
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
CSV_FIELDS = (
    "filepath", "relative_path", "filename", "label", "patient_id",
    "original_split", "new_split", "sha256",
)

PNEUMONIA_RE = re.compile(r"^(person\d+)_", re.IGNORECASE)
NORMAL_IM_RE = re.compile(r"^(IM-\d+)-", re.IGNORECASE)
NORMAL2_IM_RE = re.compile(r"^(NORMAL2-IM-\d+)-", re.IGNORECASE)

PATIENT_ID_RULES = {
    "PNEUMONIA": "Prefix matching ^(person\\d+)_ (e.g. person100_bacteria_475.jpeg -> person100).",
    "NORMAL": (
        "Prefix matching ^(IM-\\d+)- or ^(NORMAL2-IM-\\d+)-; trailing view/image tokens "
        "are excluded (e.g. IM-0011-0001-0002.jpeg -> IM-0011). The rule was selected "
        "after inspecting all NORMAL filenames."
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patient_id(filename: str, label: str) -> str | None:
    regexes = (NORMAL_IM_RE, NORMAL2_IM_RE) if label == "NORMAL" else (PNEUMONIA_RE,)
    for regex in regexes:
        match = regex.match(filename)
        if match:
            return match.group(1).lower()
    return None


def scan(dataset_root: Path, project_root: Path) -> tuple[list[dict], list[str]]:
    rows, unparsed = [], []
    for original_split in SPLITS:
        for label in LABELS:
            folder = dataset_root / original_split / label
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing required directory: {folder}")
            for path in sorted(p for p in folder.iterdir() if p.is_file()):
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                pid = patient_id(path.name, label)
                if pid is None:
                    unparsed.append(path.relative_to(project_root).as_posix())
                    continue
                rows.append({
                    "filepath": path.resolve().as_posix(),
                    "relative_path": path.relative_to(project_root).as_posix(),
                    "filename": path.name,
                    "label": label,
                    "patient_id": pid,
                    "original_split": original_split,
                    "new_split": "test" if original_split == "test" else "",
                    "sha256": sha256(path),
                })
    return rows, unparsed


def choose_validation_patients(rows: list[dict], seed: int) -> set[str]:
    """Choose patients per class, minimizing deviation from 20% of class images."""
    validation: set[str] = set()
    for label_index, label in enumerate(LABELS):
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            if row["label"] == label:
                groups[row["patient_id"]].append(row)
        items = sorted((pid, len(images)) for pid, images in groups.items())
        random.Random(seed + label_index).shuffle(items)
        target = round(sum(size for _, size in items) * VAL_RATIO)
        selected, current = [], 0
        for pid, size in items:
            if current < target:
                selected.append((pid, size))
                current += size
        if selected and abs(current - selected[-1][1] - target) <= abs(current - target):
            selected.pop()
        validation.update(pid for pid, _ in selected)
    return validation


def distribution(rows: list[dict]) -> dict:
    labels = Counter(row["label"] for row in rows)
    total = len(rows)
    return {
        "images": total,
        "patients": len({row["patient_id"] for row in rows}),
        "classes": {label: labels[label] for label in LABELS},
        "class_proportions": {
            label: round(labels[label] / total, 6) if total else 0.0 for label in LABELS
        },
    }


def validate(manifests: dict[str, list[dict]], dataset_root: Path) -> dict:
    patient_sets = {name: {r["patient_id"] for r in rows} for name, rows in manifests.items()}
    hash_sets = {name: {r["sha256"] for r in rows} for name, rows in manifests.items()}
    train_val_patient_overlap = sorted(patient_sets["train"] & patient_sets["val"])
    patient_overlaps = {
        "train_val": train_val_patient_overlap,
        "train_test": sorted(patient_sets["train"] & patient_sets["test"]),
        "val_test": sorted(patient_sets["val"] & patient_sets["test"]),
    }
    hash_overlaps = {
        "train_val": sorted(hash_sets["train"] & hash_sets["val"]),
        "train_test": sorted(hash_sets["train"] & hash_sets["test"]),
        "val_test": sorted(hash_sets["val"] & hash_sets["test"]),
    }
    source_development = {
        (dataset_root / r["original_split"] / r["label"] / r["filename"]).resolve()
        for name in ("train", "val") for r in manifests[name]
    }
    actual_development = {
        p.resolve() for split in ("train", "val") for label in LABELS
        for p in (dataset_root / split / label).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }
    referenced = [r["filepath"] for rows in manifests.values() for r in rows]
    duplicate_references = [path for path, count in Counter(referenced).items() if count != 1]
    unreadable = []
    for path_text in referenced:
        path = Path(path_text)
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            unreadable.append({"filepath": path_text, "error": f"{type(exc).__name__}: {exc}"})
    checks = {
        "train_val_patient_overlap_empty": not train_val_patient_overlap,
        "train_val_patient_overlap": train_val_patient_overlap,
        "patient_intersections": patient_overlaps,
        "all_hash_intersections_empty": not any(hash_overlaps.values()),
        "hash_intersections": hash_overlaps,
        "test_exactly_624_images": len(manifests["test"]) == 624,
        "all_files_exist_and_open": not unreadable,
        "unreadable_files": unreadable,
        "development_pool_exactly_once": source_development == actual_development and not duplicate_references,
        "duplicate_manifest_references": duplicate_references,
    }
    failed = [key for key, value in checks.items() if key in {
        "train_val_patient_overlap_empty", "all_hash_intersections_empty", "test_exactly_624_images",
        "all_files_exist_and_open", "development_pool_exactly_once",
    } and not value]
    if failed:
        raise RuntimeError(f"Split validation failed: {failed}")
    return checks


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["label"], r["patient_id"], r["filename"])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic patient-level chest X-ray splits.")
    parser.add_argument("--dataset-root", default="data/raw/chest_xray")
    parser.add_argument("--output-dir", default="data/splits/v2")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    dataset_root = (project_root / args.dataset_root).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    rows, unparsed = scan(dataset_root, project_root)
    if unparsed:
        raise RuntimeError("Unparsed patient IDs; refusing to split:\n" + "\n".join(unparsed))

    development = [r for r in rows if r["original_split"] in {"train", "val"}]
    test = [r for r in rows if r["original_split"] == "test"]
    label_by_patient: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        label_by_patient[row["patient_id"]].add(row["label"])
    multiple_label_patients = {
        pid: sorted(labels) for pid, labels in label_by_patient.items() if len(labels) > 1
    }
    if multiple_label_patients:
        raise RuntimeError(f"Patients with multiple labels: {multiple_label_patients}")

    validation_patients = choose_validation_patients(development, SEED)
    train, val = [], []
    for row in development:
        row["new_split"] = "val" if row["patient_id"] in validation_patients else "train"
        (val if row["new_split"] == "val" else train).append(row)
    manifests = {"train": train, "val": val, "test": test}
    checks = validate(manifests, dataset_root)

    patient_image_counts = Counter(r["patient_id"] for r in rows)
    image_count_distribution = Counter(patient_image_counts.values())
    summary = {
        "random_seed": SEED,
        "split_ratio": {"train": 1 - VAL_RATIO, "validation": VAL_RATIO},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "patient_id_extraction_rules": PATIENT_ID_RULES,
        "parsing": {
            "parsed_images": len(rows),
            "parsed_patients": len(patient_image_counts),
            "unparsed_count": len(unparsed),
            "unparsed_files": unparsed,
            "images_per_patient_distribution": {
                str(count): patients for count, patients in sorted(image_count_distribution.items())
            },
            "patients_with_multiple_labels": multiple_label_patients,
        },
        "splits": {name: distribution(split_rows) for name, split_rows in manifests.items()},
        "validation_checks": checks,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split_rows in manifests.items():
        write_csv(output_dir / f"{name}.csv", split_rows)
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
