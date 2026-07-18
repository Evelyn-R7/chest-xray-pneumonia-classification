"""Final test preflight checks that must not read test.csv contents or test images."""

from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from src.final_test.safety import (
    EXPECTED_PROTOCOL_SHA256,
    read_json,
    sha256_file,
    utc_now_iso,
    verify_frozen_marker,
    verify_model_hashes,
    verify_protocol_fields,
    verify_protocol_hash,
    write_json,
)


def collect_environment(project_root: str | Path) -> dict[str, Any]:
    root_usage = shutil.disk_usage("/")
    project_usage = shutil.disk_usage(project_root)
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "disk_free_root_bytes": root_usage.free,
        "disk_free_project_bytes": project_usage.free,
    }
    try:
        import tensorflow as tf

        info["tensorflow"] = tf.__version__
        info["gpu_devices"] = [device.name for device in tf.config.list_physical_devices("GPU")]
    except Exception as exc:  # pragma: no cover - environment dependent
        info["tensorflow_error"] = repr(exc)
    try:
        result = subprocess.run(["/usr/lib/wsl/lib/nvidia-smi"], capture_output=True, text=True, check=False)
        info["nvidia_smi_returncode"] = result.returncode
        info["nvidia_smi_stdout_preview"] = result.stdout[:2000]
        info["nvidia_smi_stderr_preview"] = result.stderr[:2000]
    except FileNotFoundError:
        info["nvidia_smi_error"] = "/usr/lib/wsl/lib/nvidia-smi not found"
    return info


def synthetic_model_check(model_path: str | Path) -> dict[str, Any]:
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)
    try:
        tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)
        output = np.asarray(model.predict(tensor, verbose=0)).reshape(-1)
        if output.shape != (1,):
            raise ValueError(f"Synthetic prediction shape mismatch: {output.shape}")
        if not np.all(np.isfinite(output)) or np.any((output < 0.0) | (output > 1.0)):
            raise ValueError("Synthetic prediction must be finite and in [0, 1]")
        return {"model_path": str(model_path), "output_shape": list(output.shape), "probability": float(output[0])}
    finally:
        del model
        tf.keras.backend.clear_session()


def run_preflight(
    project_root: str | Path,
    protocol_path: str | Path,
    frozen_marker_path: str | Path,
    output_dir: str | Path,
    expected_protocol_sha256: str = EXPECTED_PROTOCOL_SHA256,
) -> dict[str, Any]:
    root = Path(project_root)
    protocol_path = Path(protocol_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    actual_protocol_hash = verify_protocol_hash(protocol_path, expected_protocol_sha256)
    frozen_marker_hash = verify_frozen_marker(frozen_marker_path, expected_protocol_sha256)
    protocol = read_json(protocol_path)
    verify_protocol_fields(protocol)
    model_paths = verify_model_hashes(protocol)
    test_manifest_path = root / "data/splits/v3_clean/test.csv"
    # Preflight intentionally hashes the file only; it does not parse CSV content or open images.
    test_manifest_sha256 = sha256_file(test_manifest_path)
    synthetic = {key: synthetic_model_check(path) for key, path in model_paths.items()}
    payload = {
        "preflight_time": utc_now_iso(),
        "final_protocol_sha256": actual_protocol_hash,
        "protocol_frozen_marker_sha256": frozen_marker_hash,
        "test_manifest_sha256_file_only": test_manifest_sha256,
        "model_paths": {key: str(path) for key, path in model_paths.items()},
        "synthetic_model_checks": synthetic,
        "environment": collect_environment(root),
        "test_manifest_content_read": False,
        "test_images_read": False,
        "started_marker_created": False,
    }
    write_json(output / "preflight.json", payload)
    return payload
