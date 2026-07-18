#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$HOME/.venvs/chest-xray"
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

[[ -f "$VENV/bin/activate" ]] || fail "虚拟环境不存在：$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
configure_venv_cuda_paths
cd "$PROJECT_ROOT"

stage "GPU 前置验证"
bash scripts/verify_wsl_gpu.sh

before="$(mktemp)"
after="$(mktemp)"
trap 'rm -f "$before" "$after"' EXIT
stage "记录运行前 manifest SHA-256"
sha256sum "${MANIFESTS[@]}" | tee "$before"

stage "CNN baseline pilot（最多 3 epochs）"
python src/train_cnn.py \
  --config configs/experiments/cnn_baseline_v1.yaml \
  --max-epochs 3 \
  --run-type pilot

stage "确认 manifest 未改变"
sha256sum "${MANIFESTS[@]}" | tee "$after"
cmp -s "$before" "$after" || fail "v3_clean manifest SHA-256 发生变化"
printf 'Pilot 完成；四份 manifest 未改变。测试集未用于加载、预测或指标计算。\n'
