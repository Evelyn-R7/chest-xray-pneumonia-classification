from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

SEED = 42
VAL_RATIO = 0.20
LABELS = ("NORMAL", "PNEUMONIA")
CSV_FIELDS = (
    "filepath", "relative_path", "filename", "label", "patient_id",
    "original_split", "new_split", "sha256",
)
PATIENT_ID_RULES = {
    "PNEUMONIA": "Prefix matching ^(person\\d+)_ (normalized to lowercase).",
    "NORMAL": "Prefix matching ^(IM-\\d+)- or ^(NORMAL2-IM-\\d+)- (normalized to lowercase).",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing = set(CSV_FIELDS) - set(rows[0] if rows else ())
    if missing:
        raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["label"], r["patient_id"], r["filename"])))


def choose_validation_patients(rows: list[dict[str, str]]) -> set[str]:
    validation: set[str] = set()
    for label_index, label in enumerate(LABELS):
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            if row["label"] == label:
                groups[row["patient_id"]].append(row)
        items = sorted((pid, len(images)) for pid, images in groups.items())
        random.Random(SEED + label_index).shuffle(items)
        target = round(sum(size for _, size in items) * VAL_RATIO)
        selected: list[tuple[str, int]] = []
        current = 0
        for pid, size in items:
            if current < target:
                selected.append((pid, size))
                current += size
        if selected and abs(current - selected[-1][1] - target) <= abs(current - target):
            selected.pop()
        validation.update(pid for pid, _ in selected)
    return validation


def distribution(rows: list[dict[str, str]]) -> dict:
    counts = Counter(row["label"] for row in rows)
    total = len(rows)
    return {
        "images": total,
        "patients": len({row["patient_id"] for row in rows}),
        "classes": {label: counts[label] for label in LABELS},
        "class_proportions": {
            label: round(counts[label] / total, 6) if total else 0.0 for label in LABELS
        },
    }


def validate(
    manifests: dict[str, list[dict[str, str]]],
    excluded: list[dict[str, str]],
    remaining_development: list[dict[str, str]],
    v2_test_path: Path,
    v3_test_path: Path,
) -> dict:
    patient_sets = {name: {r["patient_id"] for r in rows} for name, rows in manifests.items()}
    hash_sets = {name: {r["sha256"] for r in rows} for name, rows in manifests.items()}
    patient_intersections = {
        "train_val": sorted(patient_sets["train"] & patient_sets["val"]),
        "train_test": sorted(patient_sets["train"] & patient_sets["test"]),
        "val_test": sorted(patient_sets["val"] & patient_sets["test"]),
    }
    hash_intersections = {
        "train_val": sorted(hash_sets["train"] & hash_sets["val"]),
        "train_test": sorted(hash_sets["train"] & hash_sets["test"]),
        "val_test": sorted(hash_sets["val"] & hash_sets["test"]),
    }
    train_val_paths = [r["filepath"] for name in ("train", "val") for r in manifests[name]]
    remaining_paths = [r["filepath"] for r in remaining_development]
    excluded_paths = {r["filepath"] for r in excluded}
    unreadable = []
    for path_text in [r["filepath"] for rows in manifests.values() for r in rows] + list(excluded_paths):
        path = Path(path_text)
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            unreadable.append({"filepath": path_text, "error": f"{type(exc).__name__}: {exc}"})
    checks = {
        "patient_intersections": patient_intersections,
        "all_patient_intersections_empty": not any(patient_intersections.values()),
        "hash_intersections": hash_intersections,
        "all_hash_intersections_empty": not any(hash_intersections.values()),
        "v3_test_matches_v2_test_sha256": file_sha256(v3_test_path) == file_sha256(v2_test_path),
        "v2_test_sha256": file_sha256(v2_test_path),
        "v3_test_sha256": file_sha256(v3_test_path),
        "excluded_absent_from_train_val": not (excluded_paths & set(train_val_paths)),
        "remaining_development_exactly_once": Counter(train_val_paths) == Counter(remaining_paths),
        "all_files_exist_and_open": not unreadable,
        "unreadable_files": unreadable,
    }
    required = (
        "all_patient_intersections_empty", "all_hash_intersections_empty",
        "v3_test_matches_v2_test_sha256", "excluded_absent_from_train_val",
        "remaining_development_exactly_once", "all_files_exist_and_open",
    )
    failed = [name for name in required if not checks[name]]
    if failed:
        raise RuntimeError(f"v3_clean validation failed: {failed}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Create patient-isolated v3_clean manifests from v2.")
    parser.add_argument("--v2-dir", default="data/splits/v2")
    parser.add_argument("--output-dir", default="data/splits/v3_clean")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    v2_dir = (project_root / args.v2_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    v2_train = read_csv(v2_dir / "train.csv")
    v2_val = read_csv(v2_dir / "val.csv")
    v2_test = read_csv(v2_dir / "test.csv")
    original_development = v2_train + v2_val
    test_patient_ids = {row["patient_id"] for row in v2_test}
    excluded = [dict(row, new_split="excluded") for row in original_development if row["patient_id"] in test_patient_ids]
    remaining = [row.copy() for row in original_development if row["patient_id"] not in test_patient_ids]

    validation_patients = choose_validation_patients(remaining)
    train, val = [], []
    for row in remaining:
        row["new_split"] = "val" if row["patient_id"] in validation_patients else "train"
        (val if row["new_split"] == "val" else train).append(row)
    manifests = {"train": train, "val": val, "test": v2_test}

    write_csv(output_dir / "train.csv", train)
    write_csv(output_dir / "val.csv", val)
    write_csv(output_dir / "excluded_from_development.csv", excluded)
    # Preserve the official benchmark manifest byte-for-byte.
    (output_dir / "test.csv").write_bytes((v2_dir / "test.csv").read_bytes())
    checks = validate(
        manifests, excluded, remaining, v2_dir / "test.csv", output_dir / "test.csv"
    )
    excluded_counts = Counter(row["label"] for row in excluded)
    summary = {
        "protocol": "patient_isolated_official_test",
        "random_seed": SEED,
        "split_ratio": {"train": 1 - VAL_RATIO, "validation": VAL_RATIO},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "patient_id_extraction_rules": PATIENT_ID_RULES,
        "original_development_pool": distribution(original_development),
        "test_patients": len(test_patient_ids),
        "excluded_overlap": {
            "patients": len({row["patient_id"] for row in excluded}),
            "images": len(excluded),
            "classes": {label: excluded_counts[label] for label in LABELS},
        },
        "remaining_development_pool": distribution(remaining),
        "splits": {name: distribution(rows) for name, rows in manifests.items()},
        "validation_checks": checks,
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
