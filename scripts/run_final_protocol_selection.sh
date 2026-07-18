#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$HOME/.venvs/chest-xray"
RESULT_DIR="$PROJECT_ROOT/results/final_protocol"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-chest-xray"
mkdir -p "$MPLCONFIGDIR"

echo "[stage 7A] Activating Python environment"
source "$VENV/bin/activate"

echo "[stage 7A] Entering project root"
cd "$PROJECT_ROOT"

manifest_files=(
  "data/splits/v3_clean/train.csv"
  "data/splits/v3_clean/val.csv"
  "data/splits/v3_clean/test.csv"
  "data/splits/v3_clean/excluded_from_development.csv"
)

before_hashes="$(mktemp)"
after_hashes="$(mktemp)"
trap 'rm -f "$before_hashes" "$after_hashes"' EXIT

echo "[stage 7A] Recording manifest SHA-256 before run"
sha256sum "${manifest_files[@]}" | tee "$before_hashes"

echo "[stage 7A] Running CPU-only, no-test-dataset pytest subset"
python -m pytest -q \
  tests/test_final_protocol.py \
  tests/test_multiseed_aggregation.py \
  tests/test_transfer_multiseed_aggregation.py \
  tests/test_imbalance_strategies.py

echo "[stage 7A] Building final validation-only protocol"
python -m src.build_final_protocol

echo "[stage 7A] Verifying required artifacts"
required_outputs=(
  "$RESULT_DIR/ensemble_validation_predictions.csv"
  "$RESULT_DIR/calibration_oof_predictions.csv"
  "$RESULT_DIR/calibration_metrics_by_fold.csv"
  "$RESULT_DIR/calibration_summary.json"
  "$RESULT_DIR/threshold_candidates.csv"
  "$RESULT_DIR/validation_metrics_final_protocol.json"
  "$RESULT_DIR/final_protocol.json"
  "$RESULT_DIR/calibration_curve_validation.png"
  "$RESULT_DIR/threshold_tradeoff_validation.png"
  "$RESULT_DIR/final_protocol_summary.md"
  "$RESULT_DIR/PROTOCOL_FROZEN"
  "$PROJECT_ROOT/reports/final_model_selection_protocol.md"
)

for path in "${required_outputs[@]}"; do
  if [[ ! -s "$path" ]]; then
    echo "[stage 7A] Missing or empty artifact: $path" >&2
    exit 1
  fi
done

echo "[stage 7A] Verifying test flags and no training flag"
python - <<'PY'
import json
from pathlib import Path

protocol = json.loads(Path("results/final_protocol/final_protocol.json").read_text(encoding="utf-8"))
if protocol.get("test_loaded") is not False:
    raise SystemExit("test_loaded is not false")
if protocol.get("test_evaluated") is not False:
    raise SystemExit("test_evaluated is not false")
if protocol.get("training_performed") is not False:
    raise SystemExit("training_performed is not false")
if len(protocol.get("best_model_sha256", {})) != 3:
    raise SystemExit("Expected three model SHA-256 entries")
PY

echo "[stage 7A] Recording manifest SHA-256 after run"
sha256sum "${manifest_files[@]}" | tee "$after_hashes"

echo "[stage 7A] Verifying manifests are unchanged"
diff -u "$before_hashes" "$after_hashes"

echo "[stage 7A] Complete. No GPU validation, test image loading, or training was performed."
