#!/usr/bin/env bash
set -euo pipefail

source "$HOME/.venvs/chest-xray/bin/activate"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTROL_REGISTRY="results/experiments/efficientnetb0_transfer_v1/multiseed_registry.json"
SEEDS=(42 2025 2026)
MANIFESTS=(
  data/splits/v3_clean/train.csv
  data/splits/v3_clean/val.csv
  data/splits/v3_clean/test.csv
  data/splits/v3_clean/excluded_from_development.csv
)
ARTIFACTS=(
  resolved_config.yaml environment.json model_summary.txt trainable_layers_phase1.txt
  trainable_layers_phase2.txt phase1_history.csv phase2_history.csv combined_history.csv
  phase1_best.keras phase2_best.keras best_model.keras val_predictions.csv val_metrics.json
  run_summary.md learning_curves.png confusion_matrix_val.png roc_curve_val.png pr_curve_val.png
  console.log
)

configure_cuda_library_path() {
  local nvidia_root="$HOME/.venvs/chest-xray/lib/python3.11/site-packages/nvidia" nvidia_libs
  nvidia_libs="$(find "$nvidia_root" -mindepth 2 -maxdepth 2 -type d -name lib -print | sort | paste -sd: -)"
  [[ -n "$nvidia_libs" ]] || { echo "ERROR: NVIDIA pip libraries not found under $nvidia_root" >&2; return 1; }
  export LD_LIBRARY_PATH="$nvidia_libs:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  echo "Configured CUDA library path for TensorFlow training process."
}

require_training_gpu_visible() {
  python - <<'PY'
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("Training-process GPU devices:", gpus)
if not gpus:
    raise SystemExit("ERROR: TensorFlow training process cannot see a GPU")
PY
}

require_complete_run() {
  local run_dir="$1" expected_experiment="$2" expected_seed="$3"
  python - "$run_dir" "$expected_experiment" "$expected_seed" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

run_dir = Path(sys.argv[1])
expected_experiment = sys.argv[2]
expected_seed = int(sys.argv[3])
artifacts = [
    "resolved_config.yaml", "environment.json", "model_summary.txt",
    "trainable_layers_phase1.txt", "trainable_layers_phase2.txt",
    "phase1_history.csv", "phase2_history.csv", "combined_history.csv",
    "phase1_best.keras", "phase2_best.keras", "best_model.keras",
    "val_predictions.csv", "val_metrics.json", "run_summary.md",
    "learning_curves.png", "confusion_matrix_val.png", "roc_curve_val.png",
    "pr_curve_val.png", "console.log",
]
missing = [name for name in artifacts if not (run_dir / name).is_file() or (run_dir / name).stat().st_size == 0]
if missing:
    raise SystemExit(f"ERROR: incomplete run {run_dir}; missing/empty: {missing}")
config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
if config.get("experiment_name") != expected_experiment or int(config.get("seed")) != expected_seed:
    raise SystemExit("ERROR: run experiment or seed mismatch")
if config.get("model_name") != "efficientnetb0" or config.get("data_config") != "configs/data_v3_clean.yaml":
    raise SystemExit("ERROR: run config is not EfficientNetB0 v3_clean")
if float(config.get("threshold")) != 0.5 or int(config.get("batch_size")) != 16:
    raise SystemExit("ERROR: threshold or batch size changed")
if bool(config.get("mixed_precision")):
    raise SystemExit("ERROR: mixed precision is forbidden")
if int(config.get("data_num_parallel_calls")) != 1 or bool(config.get("data_prefetch")):
    raise SystemExit("ERROR: WSL serial read settings are not active")

predictions = pd.read_csv(run_dir / "val_predictions.csv")
if len(predictions) != 954:
    raise SystemExit(f"ERROR: expected 954 validation predictions, found {len(predictions)}")
metrics = json.loads((run_dir / "val_metrics.json").read_text(encoding="utf-8"))
if not metrics or not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in metrics.values()):
    raise SystemExit("ERROR: validation metrics contain missing or non-finite values")

phase1 = pd.read_csv(run_dir / "phase1_history.csv")
phase2 = pd.read_csv(run_dir / "phase2_history.csv")
combined = pd.read_csv(run_dir / "combined_history.csv")
if phase1.empty or phase2.empty or len(combined) != len(phase1) + len(phase2):
    raise SystemExit("ERROR: phase histories do not prove both phases executed")
for frame_name, frame in {"phase1": phase1, "phase2": phase2, "combined": combined}.items():
    numeric = frame.select_dtypes(include="number").to_numpy(dtype=float)
    if numeric.size == 0 or not math.isfinite(float(numeric.sum())):
        raise SystemExit(f"ERROR: {frame_name} history contains non-finite values")

summary = (run_dir / "run_summary.md").read_text(encoding="utf-8")
if not re.search(r"Best phase: phase[12]", summary):
    raise SystemExit("ERROR: best checkpoint phase is not recorded")
log = (run_dir / "console.log").read_text(encoding="utf-8", errors="replace")
for pattern in (r"\bnan\b", r"\binf\b", "out of memory", "resource_exhausted", "traceback", "training failed"):
    if re.search(pattern, log, re.IGNORECASE):
        raise SystemExit(f"ERROR: suspicious training log pattern: {pattern}")
for path in run_dir.iterdir():
    lower = path.name.lower()
    if "test" in lower and ("prediction" in lower or "metric" in lower):
        raise SystemExit(f"ERROR: forbidden test output: {path}")
PY
}

find_existing_complete_run() {
  local experiment="$1" seed="$2" root="results/experiments/$experiment/seed_$seed"
  [[ -d "$root" ]] || return 1
  mapfile -t candidates < <(find "$root" -mindepth 2 -maxdepth 2 -name resolved_config.yaml -printf '%h\n' | sort)
  local complete=()
  for candidate in "${candidates[@]}"; do
    if require_complete_run "$candidate" "$experiment" "$seed" >/dev/null 2>&1; then
      complete+=("$candidate")
    fi
  done
  if [[ "${#complete[@]}" -gt 1 ]]; then
    printf 'ERROR: multiple complete runs exist for %s seed %s:\n' "$experiment" "$seed" >&2
    printf '  %s\n' "${complete[@]}" >&2
    return 2
  fi
  [[ "${#complete[@]}" -eq 1 ]] || return 1
  printf '%s\n' "${complete[0]}"
}

run_one() {
  local experiment="$1" config="$2" seed="$3" existing log status run_dir allocated_run_dir
  if existing="$(find_existing_complete_run "$experiment" "$seed")"; then
    echo "Reusing complete existing run for $experiment seed $seed: $existing" >&2
    printf '%s\n' "$existing"
    return 0
  fi
  log="$(mktemp)"
  set +e
  python src/train_transfer.py --config "$config" --seed "$seed" --run-type full 2>&1 | tee "$log" >&2
  status="${PIPESTATUS[0]}"
  set -e
  allocated_run_dir="$(grep -a '^ALLOCATED_RUN_DIR=' "$log" | tail -n 1 | cut -d= -f2-)"
  if [[ -n "$allocated_run_dir" && -d "$allocated_run_dir" ]]; then
    cp "$log" "$allocated_run_dir/console.log"
  fi
  if [[ "$status" -ne 0 ]]; then
    echo "ERROR: $experiment seed $seed training failed with exit code $status" >&2
    [[ -n "$allocated_run_dir" ]] && echo "Failure log: $allocated_run_dir/console.log" >&2
    rm -f "$log"
    return "$status"
  fi
  run_dir="$(grep -a '^RUN_DIR=' "$log" | tail -n 1 | cut -d= -f2-)"
  rm -f "$log"
  [[ -n "$run_dir" && -d "$run_dir" ]] || { echo "ERROR: invalid RUN_DIR for $experiment seed $seed" >&2; return 1; }
  require_complete_run "$run_dir" "$experiment" "$seed"
  printf '%s\n' "$run_dir"
}

write_registry() {
  local experiment="$1" registry="$2" seed42="$3" seed2025="$4" seed2026="$5" hashes_file="$6"
  python - "$experiment" "$registry" "$seed42" "$seed2025" "$seed2026" "$hashes_file" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

experiment, registry_path, seed42, seed2025, seed2026, hashes_file = sys.argv[1:]
manifest_sha256 = {}
for line in Path(hashes_file).read_text(encoding="utf-8").splitlines():
    digest, path = line.split(maxsplit=1)
    manifest_sha256[path] = digest
runs = {42: seed42, 2025: seed2025, 2026: seed2026}
payload = {
    "experiment_name": experiment,
    "model_name": "efficientnetb0",
    "data_protocol": "v3_clean_train_val_only",
    "fixed_threshold": 0.5,
    "seed_42_run_dir": seed42,
    "seed_2025_run_dir": seed2025,
    "seed_2026_run_dir": seed2026,
    "runs": {
        f"seed_{seed}": {
            "run_dir": run_dir,
            "resolved_config": f"{run_dir}/resolved_config.yaml",
            "val_metrics": f"{run_dir}/val_metrics.json",
            "val_predictions": f"{run_dir}/val_predictions.csv",
            "combined_history": f"{run_dir}/combined_history.csv",
        }
        for seed, run_dir in runs.items()
    },
    "manifest_sha256": manifest_sha256,
    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
Path(registry_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

configure_cuda_library_path
./scripts/verify_wsl_gpu.sh
require_training_gpu_visible

python src/aggregate_transfer_multiseed.py --registry "$CONTROL_REGISTRY" >/dev/null

before="$(mktemp)"
after="$(mktemp)"
trap 'rm -f "$before" "$after"' EXIT
sha256sum "${MANIFESTS[@]}" | tee "$before"

cw42="$(run_one efficientnetb0_class_weight_v1 configs/experiments/efficientnetb0_class_weight_v1.yaml 42)"
cw2025="$(run_one efficientnetb0_class_weight_v1 configs/experiments/efficientnetb0_class_weight_v1.yaml 2025)"
cw2026="$(run_one efficientnetb0_class_weight_v1 configs/experiments/efficientnetb0_class_weight_v1.yaml 2026)"
focal42="$(run_one efficientnetb0_focal_v1 configs/experiments/efficientnetb0_focal_v1.yaml 42)"
focal2025="$(run_one efficientnetb0_focal_v1 configs/experiments/efficientnetb0_focal_v1.yaml 2025)"
focal2026="$(run_one efficientnetb0_focal_v1 configs/experiments/efficientnetb0_focal_v1.yaml 2026)"

sha256sum "${MANIFESTS[@]}" | tee "$after"
cmp -s "$before" "$after" || { echo "ERROR: manifest hashes changed" >&2; exit 1; }

write_registry efficientnetb0_class_weight_v1 results/experiments/efficientnetb0_class_weight_v1/multiseed_registry.json "$cw42" "$cw2025" "$cw2026" "$after"
write_registry efficientnetb0_focal_v1 results/experiments/efficientnetb0_focal_v1/multiseed_registry.json "$focal42" "$focal2025" "$focal2026" "$after"

echo "CLASS_WEIGHT_REGISTRY=$PROJECT_ROOT/results/experiments/efficientnetb0_class_weight_v1/multiseed_registry.json"
echo "FOCAL_REGISTRY=$PROJECT_ROOT/results/experiments/efficientnetb0_focal_v1/multiseed_registry.json"
