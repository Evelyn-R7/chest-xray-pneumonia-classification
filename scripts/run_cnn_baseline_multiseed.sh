#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$HOME/.venvs/chest-xray"
EXPERIMENT_ROOT="$PROJECT_ROOT/results/experiments/cnn_baseline_v1"
REGISTRY="$EXPERIMENT_ROOT/multiseed_registry.json"
MANIFESTS=(
  "data/splits/v3_clean/train.csv"
  "data/splits/v3_clean/val.csv"
  "data/splits/v3_clean/test.csv"
  "data/splits/v3_clean/excluded_from_development.csv"
)
stage() { printf '\n===== %s =====\n' "$1"; }
fail() { printf 'ERROR: %s\n' "$1" >&2; exit 1; }
configure_venv_cuda_paths() {
  local root="$VENV/lib/python3.11/site-packages/nvidia" libs
  libs="$(find "$root" -mindepth 2 -maxdepth 2 -type d -name lib -print | sort | paste -sd: -)"
  [[ -n "$libs" ]] || fail "未找到虚拟环境 NVIDIA 动态库"
  export LD_LIBRARY_PATH="$libs:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}
validate_run() {
  local run_dir="$1" seed="$2"
  python - "$run_dir" "$seed" <<'PY'
import sys
from src.aggregate_multiseed import validate_complete_run

validated = validate_complete_run(sys.argv[1], expected_seed=int(sys.argv[2]))
print(f"Validated seed {validated['config']['seed']}: {validated['run_dir']}")
PY
}
run_seed() {
  local seed="$1" log_file="$2" run_dir
  stage "训练 seed $seed"
  python src/train_cnn.py \
    --config configs/experiments/cnn_baseline_v1.yaml \
    --seed "$seed" \
    --run-type full 2>&1 | tee "$log_file"
  run_dir="$(grep -a '^RUN_DIR=' "$log_file" | tail -n 1 | cut -d= -f2-)"
  [[ -n "$run_dir" && -d "$run_dir" ]] || fail "seed $seed 未输出有效 RUN_DIR"
  validate_run "$run_dir" "$seed"
  cp "$log_file" "$run_dir/console.log"
  printf '%s\n' "$run_dir"
}

[[ -f "$VENV/bin/activate" ]] || fail "虚拟环境不存在：$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
configure_venv_cuda_paths
cd "$PROJECT_ROOT"

stage "GPU 前置验证"
bash scripts/verify_wsl_gpu.sh

before="$(mktemp)"
after="$(mktemp)"
log_2025="$(mktemp)"
log_2026="$(mktemp)"
trap 'rm -f "$before" "$after" "$log_2025" "$log_2026"' EXIT

stage "记录运行前 manifest SHA-256"
sha256sum "${MANIFESTS[@]}" | tee "$before"

stage "唯一识别现有 seed 42 full run"
seed_42_dir="$(python - "$EXPERIMENT_ROOT/seed_42" <<'PY'
import sys
from src.aggregate_multiseed import find_unique_seed42_full
print(find_unique_seed42_full(sys.argv[1]))
PY
)"
printf 'seed 42 full run: %s\n' "$seed_42_dir"

seed_2025_dir="$(run_seed 2025 "$log_2025" | tee /dev/stderr | tail -n 1)"
seed_2026_dir="$(run_seed 2026 "$log_2026" | tee /dev/stderr | tail -n 1)"

stage "确认 manifest 未改变"
sha256sum "${MANIFESTS[@]}" | tee "$after"
cmp -s "$before" "$after" || fail "v3_clean manifest SHA-256 发生变化"

stage "写入明确路径的多种子登记文件"
python - "$REGISTRY" "$seed_42_dir" "$seed_2025_dir" "$seed_2026_dir" "$before" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

registry_path = Path(sys.argv[1]).resolve()
run_dirs = {seed: Path(path).resolve() for seed, path in zip((42, 2025, 2026), sys.argv[2:5])}
hashes = {}
for line in Path(sys.argv[5]).read_text(encoding="utf-8").splitlines():
    digest, filename = line.split(maxsplit=1)
    hashes[filename] = digest
registry = {
    "experiment_name": "cnn_baseline_v1",
    "data_protocol": "configs/data_v3_clean.yaml (train/validation only)",
    "fixed_threshold": 0.5,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "manifest_sha256": hashes,
}
for seed, run_dir in run_dirs.items():
    registry[f"seed_{seed}_run_dir"] = str(run_dir)
    registry[f"seed_{seed}_resolved_config"] = str(run_dir / "resolved_config.yaml")
    registry[f"seed_{seed}_val_metrics"] = str(run_dir / "val_metrics.json")
    registry[f"seed_{seed}_val_predictions"] = str(run_dir / "val_predictions.csv")
registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
print(f"REGISTRY={registry_path}")
PY

printf '多种子训练与登记完成。seed 42 未重新运行；测试集未用于加载、预测或指标计算。\n'
