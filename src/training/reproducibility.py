"""Reproducibility and environment capture helpers."""

from __future__ import annotations

import platform
import random
from typing import Any

import numpy as np
import tensorflow as tf


def configure_reproducibility(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    deterministic = False
    determinism_error = None
    try:
        tf.config.experimental.enable_op_determinism()
        deterministic = True
    except Exception as exc:  # Availability varies by TensorFlow/platform.
        determinism_error = f"{type(exc).__name__}: {exc}"

    gpus = tf.config.list_physical_devices("GPU")
    memory_growth = {}
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            memory_growth[gpu.name] = True
        except RuntimeError as exc:
            memory_growth[gpu.name] = f"not changed: {exc}"

    build_info = dict(tf.sysconfig.get_build_info())
    return {
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "gpu_devices": [gpu.name for gpu in gpus],
        "cuda_build_version": build_info.get("cuda_version"),
        "cudnn_build_version": build_info.get("cudnn_version"),
        "tensorflow_build_info": build_info,
        "seed": seed,
        "deterministic_ops_enabled": deterministic,
        "determinism_error": determinism_error,
        "memory_growth": memory_growth,
    }
