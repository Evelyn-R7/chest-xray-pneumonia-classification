"""Safety checks for the one-time final test evaluation."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED_PROTOCOL_SHA256 = "6835e0135b286046c37d2fa765aeaea2f564e3b8155e3012c81052088b75885e"
CONFIRMATION_STRING = "I_UNDERSTAND_THIS_IS_THE_FINAL_TEST_EVALUATION"
EXPECTED_MODEL_SHA256 = {
    "seed_42": "113e48ecabae1790df03347848852f7aef991454561c191e225f7dc74ddf1fd8",
    "seed_2025": "7da554ecdabef9888216c3c6d9c5f455df6eeb75282ec9e817d225fd90df34dc",
    "seed_2026": "f1be6eda983215ea9c836aa73a82a0eae50b93cfa1afbf0d7ef24cb01967186a",
}
EXPECTED_BALANCED_THRESHOLD = 0.5618644666666667
BENCHMARK_THRESHOLD = 0.5
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260718


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_protocol_hash(protocol_path: str | Path, expected: str = EXPECTED_PROTOCOL_SHA256) -> str:
    actual = sha256_file(protocol_path)
    if actual != expected:
        raise ValueError(f"final_protocol.json SHA-256 mismatch: expected {expected}, found {actual}")
    return actual


def verify_confirmation(expected_protocol_sha256: str, confirmation: str) -> None:
    if expected_protocol_sha256 != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("Incorrect --expected-protocol-sha256")
    if confirmation != CONFIRMATION_STRING:
        raise ValueError("Incorrect --confirm-one-time-test value")


def verify_frozen_marker(marker_path: str | Path, expected_protocol_sha256: str = EXPECTED_PROTOCOL_SHA256) -> str:
    marker = Path(marker_path)
    text = marker.read_text(encoding="utf-8")
    expected_line = f"final_protocol_json_sha256: {expected_protocol_sha256}"
    if expected_line not in text:
        raise ValueError("PROTOCOL_FROZEN does not record the expected final_protocol.json hash")
    return sha256_file(marker)


def verify_protocol_fields(protocol: dict[str, Any]) -> None:
    checks = {
        "test_loaded": False,
        "test_evaluated": False,
        "training_performed": False,
        "selected_calibration_method": "none",
        "benchmark_threshold": BENCHMARK_THRESHOLD,
        "balanced_threshold": EXPECTED_BALANCED_THRESHOLD,
    }
    for key, expected in checks.items():
        actual = protocol.get(key)
        if actual != expected:
            raise ValueError(f"Protocol field {key!r} mismatch: expected {expected!r}, found {actual!r}")


def verify_model_hashes(protocol: dict[str, Any]) -> dict[str, Path]:
    paths = protocol.get("best_model_paths")
    hashes = protocol.get("best_model_sha256")
    if not isinstance(paths, dict) or not isinstance(hashes, dict):
        raise ValueError("Protocol must contain model paths and SHA-256 mappings")
    resolved: dict[str, Path] = {}
    for key, expected_hash in EXPECTED_MODEL_SHA256.items():
        if hashes.get(key) != expected_hash:
            raise ValueError(f"Frozen protocol model hash mismatch for {key}")
        path = Path(paths[key])
        if not path.is_file():
            raise FileNotFoundError(f"Model file missing for {key}: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"On-disk model hash mismatch for {key}: expected {expected_hash}, found {actual_hash}")
        resolved[key] = path
    return resolved


def check_one_time_markers(final_test_dir: str | Path) -> None:
    root = Path(final_test_dir)
    if (root / "TEST_EVALUATED").exists():
        raise FileExistsError("TEST_EVALUATED already exists; final test evaluation cannot be repeated")
    if (root / "TEST_EVALUATION_STARTED").exists():
        raise FileExistsError("TEST_EVALUATION_STARTED exists without TEST_EVALUATED; audit previous interruption before retry")


def create_started_marker(final_test_dir: str | Path, payload: dict[str, Any]) -> Path:
    root = Path(final_test_dir)
    root.mkdir(parents=True, exist_ok=True)
    started = root / "TEST_EVALUATION_STARTED"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    with os.fdopen(os.open(started, flags, 0o644), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return started


def create_evaluated_marker(final_test_dir: str | Path, payload: dict[str, Any]) -> Path:
    evaluated = Path(final_test_dir) / "TEST_EVALUATED"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    with os.fdopen(os.open(evaluated, flags, 0o644), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return evaluated
