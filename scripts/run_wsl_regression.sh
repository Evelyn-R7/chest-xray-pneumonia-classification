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
  local nvidia_root="$VENV/lib/python3.11/site-packages/nvidia"
  local nvidia_libs
  [[ -d "$nvidia_root" ]] || fail "虚拟环境 NVIDIA 包目录不存在：$nvidia_root"
  nvidia_libs="$(find "$nvidia_root" -mindepth 2 -maxdepth 2 -type d -name lib -print | sort | paste -sd: -)"
  [[ -n "$nvidia_libs" ]] || fail "未找到虚拟环境内的 NVIDIA 动态库目录"
  export LD_LIBRARY_PATH="$nvidia_libs:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

[[ -f "$VENV/bin/activate" ]] || fail "虚拟环境不存在：$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
configure_venv_cuda_paths
cd "$PROJECT_ROOT"

stage "TensorFlow GPU 前置验证"
bash "$PROJECT_ROOT/scripts/verify_wsl_gpu.sh"

stage "记录运行前 manifest SHA-256"
before_file="$(mktemp)"
after_file="$(mktemp)"
pipeline_log="$(mktemp)"
pytest_log="$(mktemp)"
smoke_log="$(mktemp)"
cleanup() { rm -f "$before_file" "$after_file" "$pipeline_log" "$pytest_log" "$smoke_log"; }
trap cleanup EXIT
for file in "${MANIFESTS[@]}"; do [[ -f "$file" ]] || fail "manifest 不存在：$file"; done
sha256sum "${MANIFESTS[@]}" | tee "$before_file"

stage "数据管道回归"
python src/check_data_pipeline.py | tee "$pipeline_log"
python - <<'PY'
import json
from pathlib import Path

result = json.loads(Path("reports/data_pipeline/check_results.json").read_text(encoding="utf-8"))
actual = result.get("traversed_samples")
expected = {"train": 3821, "val": 954, "test": 624}
if actual != expected:
    raise RuntimeError(f"遍历数量不符合预期：actual={actual}, expected={expected}")
print("Traversal counts verified:", actual)
PY

stage "pytest 回归"
python -m pytest -q | tee "$pytest_log"
python - "$pytest_log" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
matches = re.findall(r"(\d+) passed", text)
if not matches or int(matches[-1]) < 14:
    raise RuntimeError(f"pytest 通过数少于 14，或无法解析结果：{text[-1000:]}")
print(f"pytest passed count verified: {matches[-1]}")
PY

stage "Smoke Test（仅 2 个训练 step，不保存权重）"
python src/smoke_test.py | tee "$smoke_log"
grep -Eq "['\"]status['\"]: ['\"]success['\"]" "$smoke_log" \
  || fail "Smoke Test 未输出成功状态"

stage "比较运行后 manifest SHA-256"
sha256sum "${MANIFESTS[@]}" | tee "$after_file"
if ! cmp -s "$before_file" "$after_file"; then
  printf 'ERROR: manifest SHA-256 已发生变化：\n' >&2
  diff -u "$before_file" "$after_file" || true
  exit 1
fi

stage "阶段 3 回归验证成功"
printf '遍历数量 train=3821, val=954, test=624；pytest 至少 14 项通过；Smoke Test 成功。\n'
printf '四份 v3_clean manifest 的 SHA-256 保持不变。未保存模型权重，未进行正式训练。\n'
