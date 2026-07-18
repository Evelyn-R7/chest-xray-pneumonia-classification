#!/usr/bin/env bash
# Shared implementation; source only after set -euo pipefail and venv activation.

run_transfer_experiment() {
  local config="$1" run_type="$2" project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  local before after log run_dir allocated_run_dir train_status
  local manifests=(
    data/splits/v3_clean/train.csv data/splits/v3_clean/val.csv
    data/splits/v3_clean/test.csv data/splits/v3_clean/excluded_from_development.csv
  )
  local artifacts=(
    resolved_config.yaml environment.json model_summary.txt trainable_layers_phase1.txt
    trainable_layers_phase2.txt phase1_history.csv phase2_history.csv combined_history.csv
    phase1_best.keras phase2_best.keras best_model.keras val_predictions.csv val_metrics.json
    run_summary.md learning_curves.png confusion_matrix_val.png roc_curve_val.png pr_curve_val.png
  )
  cd "$project_root"
  local weight_file
  case "$config" in
    *vgg16*) weight_file="$HOME/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5" ;;
    *efficientnetb0*) weight_file="$HOME/.keras/models/efficientnetb0_notop.h5" ;;
    *) echo "ERROR: unknown transfer config: $config" >&2; return 1 ;;
  esac
  [[ -s "$weight_file" ]] || {
    echo "ERROR: ImageNet weight file is missing. Run:" >&2
    case "$config" in
      *vgg16*) echo "  ./scripts/download_transfer_weights.sh vgg16" >&2 ;;
      *) echo "  ./scripts/download_transfer_weights.sh efficientnetb0" >&2 ;;
    esac
    return 1
  }
  local nvidia_root="$HOME/.venvs/chest-xray/lib/python3.11/site-packages/nvidia" nvidia_libs
  nvidia_libs="$(find "$nvidia_root" -mindepth 2 -maxdepth 2 -type d -name lib -print | sort | paste -sd: -)"
  [[ -n "$nvidia_libs" ]] || { echo "ERROR: NVIDIA pip libraries not found" >&2; return 1; }
  export LD_LIBRARY_PATH="$nvidia_libs:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  ./scripts/verify_wsl_gpu.sh
  before="$(mktemp)"; after="$(mktemp)"; log="$(mktemp)"
  trap 'rm -f "$before" "$after" "$log"' RETURN
  sha256sum "${manifests[@]}" | tee "$before"
  set +e
  python src/train_transfer.py --config "$config" --run-type "$run_type" 2>&1 | tee "$log"
  train_status="${PIPESTATUS[0]}"
  set -e
  allocated_run_dir="$(grep -a '^ALLOCATED_RUN_DIR=' "$log" | tail -n 1 | cut -d= -f2-)"
  if [[ -n "$allocated_run_dir" && -d "$allocated_run_dir" ]]; then
    cp "$log" "$allocated_run_dir/console.log"
  fi
  if [[ "$train_status" -ne 0 ]]; then
    echo "ERROR: transfer training failed with exit code $train_status" >&2
    [[ -n "$allocated_run_dir" ]] && echo "Failure log: $allocated_run_dir/console.log" >&2
    return "$train_status"
  fi
  run_dir="$(grep -a '^RUN_DIR=' "$log" | tail -n 1 | cut -d= -f2-)"
  [[ -n "$run_dir" && -d "$run_dir" ]] || { echo "ERROR: invalid RUN_DIR" >&2; return 1; }
  cp "$log" "$run_dir/console.log"
  for artifact in "${artifacts[@]}"; do
    [[ -s "$run_dir/$artifact" ]] || { echo "ERROR: missing/empty $artifact" >&2; return 1; }
  done
  [[ "$(wc -l < "$run_dir/val_predictions.csv")" -eq 955 ]] \
    || { echo "ERROR: val_predictions.csv must contain 954 predictions" >&2; return 1; }
  python - "$run_dir" <<'PY'
import json, math, sys
from pathlib import Path
root = Path(sys.argv[1])
metrics = json.loads((root / "val_metrics.json").read_text(encoding="utf-8"))
if not metrics or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in metrics.values()):
    raise SystemExit("ERROR: validation metrics contain missing/non-finite values")
for path in root.iterdir():
    lower = path.name.lower()
    if "test" in lower and ("prediction" in lower or "metric" in lower):
        raise SystemExit(f"ERROR: forbidden test output: {path}")
PY
  sha256sum "${manifests[@]}" | tee "$after"
  cmp -s "$before" "$after" || { echo "ERROR: manifest hashes changed" >&2; return 1; }
  echo "Validated transfer run: $run_dir"
}
