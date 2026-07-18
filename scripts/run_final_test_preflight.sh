#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$HOME/.venvs/chest-xray"
EXPECTED_PROTOCOL_SHA256="6835e0135b286046c37d2fa765aeaea2f564e3b8155e3012c81052088b75885e"
PREFLIGHT_DIR="$PROJECT_ROOT/results/final_test_preflight"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-chest-xray"
mkdir -p "$MPLCONFIGDIR" "$PREFLIGHT_DIR"

source "$VENV/bin/activate"
cd "$PROJECT_ROOT"

configure_nvidia_pip_libraries() {
  local nvidia_root="$VENV/lib/python3.11/site-packages/nvidia"
  local nvidia_libs
  nvidia_libs="$(find "$nvidia_root" -mindepth 2 -maxdepth 2 -type d -name lib -print | sort | paste -sd: -)"
  if [[ -z "$nvidia_libs" ]]; then
    echo "ERROR: NVIDIA pip libraries not found under $nvidia_root" >&2
    exit 1
  fi
  export LD_LIBRARY_PATH="$nvidia_libs:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

configure_nvidia_pip_libraries

echo "[final test preflight] GPU/TensorFlow environment check"
python - <<'PY'
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPU devices:", tf.config.list_physical_devices("GPU"))
PY

echo "[final test preflight] Synthetic final-test unit tests"
python -m pytest -q tests/test_final_test_evaluation.py

echo "[final test preflight] Running preflight without reading test.csv contents or test images"
python -m src.run_final_test_preflight \
  --expected-protocol-sha256 "$EXPECTED_PROTOCOL_SHA256" \
  2>&1 | tee "$PREFLIGHT_DIR/console.log"

echo "[final test preflight] Complete. STARTED/EVALUATED markers were not created."
