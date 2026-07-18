#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$HOME/.venvs/chest-xray"
EXPECTED_PROTOCOL_SHA256="6835e0135b286046c37d2fa765aeaea2f564e3b8155e3012c81052088b75885e"
CONFIRMATION="I_UNDERSTAND_THIS_IS_THE_FINAL_TEST_EVALUATION"
FINAL_TEST_DIR="$PROJECT_ROOT/results/final_test"
PREFLIGHT_JSON="$PROJECT_ROOT/results/final_test_preflight/preflight.json"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-chest-xray"
mkdir -p "$MPLCONFIGDIR" "$FINAL_TEST_DIR"

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

if [[ ! -s "$PREFLIGHT_JSON" ]]; then
  echo "Missing successful preflight: $PREFLIGHT_JSON" >&2
  exit 1
fi

actual_hash="$(sha256sum results/final_protocol/final_protocol.json | awk '{print $1}')"
if [[ "$actual_hash" != "$EXPECTED_PROTOCOL_SHA256" ]]; then
  echo "final_protocol.json hash mismatch: $actual_hash" >&2
  exit 1
fi

python - <<'PY'
import json
from pathlib import Path
from src.final_test.safety import read_json, verify_model_hashes

protocol = read_json("results/final_protocol/final_protocol.json")
verify_model_hashes(protocol)
PY

if [[ -e "$FINAL_TEST_DIR/TEST_EVALUATION_STARTED" || -e "$FINAL_TEST_DIR/TEST_EVALUATED" ]]; then
  echo "Final test one-time marker already exists; refusing to run." >&2
  ls -la "$FINAL_TEST_DIR" >&2
  exit 1
fi

echo
echo "============================================================"
echo "THIS WILL LOAD AND EVALUATE THE FINAL TEST SET ONCE."
echo "MODEL, CALIBRATION, AND THRESHOLDS ARE FROZEN."
echo "Type exactly: FINAL TEST"
echo "============================================================"
read -r user_confirmation
if [[ "$user_confirmation" != "FINAL TEST" ]]; then
  echo "Confirmation did not match; exiting without evaluation." >&2
  exit 1
fi

console_log="$FINAL_TEST_DIR/final_test_once_console.log"
python -m src.evaluate_final_test_once \
  --expected-protocol-sha256 "$EXPECTED_PROTOCOL_SHA256" \
  --confirm-one-time-test "$CONFIRMATION" \
  2>&1 | tee "$console_log"

if [[ ! -s "$FINAL_TEST_DIR/TEST_EVALUATED" ]]; then
  echo "TEST_EVALUATED marker was not created." >&2
  exit 1
fi

echo "Final test evaluation completed once. Do not rerun."
