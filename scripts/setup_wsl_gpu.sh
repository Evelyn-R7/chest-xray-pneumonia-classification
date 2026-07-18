#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$HOME/.venvs/chest-xray"
PIP_NETWORK_ARGS=(--timeout 120 --retries 20 --resume-retries 50)

stage() { printf '\n===== %s =====\n' "$1"; }
fail() { printf 'ERROR: %s\n' "$1" >&2; exit 1; }
pip_install_with_retries() {
  local attempt max_attempts=10
  for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    printf 'pip 安装尝试 %d/%d：%s\n' "$attempt" "$max_attempts" "$*"
    if python -m pip install "${PIP_NETWORK_ARGS[@]}" "$@"; then
      return 0
    fi
    if (( attempt < max_attempts )); then
      printf 'WARNING: 本次下载失败，5 秒后仅重试当前安装项。\n' >&2
      sleep 5
    fi
  done
  fail "pip 安装连续 ${max_attempts} 次失败：$*"
}

stage "检查 WSL GPU 链路"
[[ -x /usr/lib/wsl/lib/nvidia-smi ]] || fail "/usr/lib/wsl/lib/nvidia-smi 不存在或不可执行"
[[ -e /dev/dxg ]] || fail "/dev/dxg 不存在"
[[ -f /usr/lib/wsl/lib/libcuda.so ]] || fail "/usr/lib/wsl/lib/libcuda.so 不存在"
/usr/lib/wsl/lib/nvidia-smi

stage "检查项目和下载工具"
[[ -d "$PROJECT_ROOT" ]] || fail "项目目录不存在：$PROJECT_ROOT"
[[ -f "$PROJECT_ROOT/requirements.txt" ]] || fail "requirements.txt 不存在"
command -v curl >/dev/null || fail "curl 未安装；请先人工安装 curl 后重试"

stage "准备 uv"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  printf '未检测到 uv，使用 Astral 官方安装脚本。\n'
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null || fail "uv 安装后仍无法在 PATH 中找到"
uv --version

stage "安装 Python 3.11 并创建独立虚拟环境"
uv python install 3.11
mkdir -p "$HOME/.venvs"
if [[ -e "$VENV" ]]; then
  printf '虚拟环境已存在，将复用：%s\n' "$VENV"
else
  uv venv --python 3.11 --seed "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

stage "验证 Python 环境"
which python
python --version
python -m pip --version
[[ "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "3.11" ]] \
  || fail "虚拟环境 Python 不是 3.11"
[[ "$(command -v python)" == "$VENV/bin/python" ]] \
  || fail "当前 Python 不属于 $VENV"

stage "安装 TensorFlow GPU 环境和项目依赖"
printf '大文件下载启用延长超时、连接重试和断点续传；网络中断后可安全重跑本脚本。\n'
pip_install_with_retries --upgrade pip setuptools wheel
pip_install_with_retries "tensorflow[and-cuda]==2.21.*"
for package in pandas numpy Pillow PyYAML matplotlib pytest scikit-learn; do
  pip_install_with_retries "$package"
done

stage "检查 requirements.txt 的 TensorFlow 约束"
if python - "$PROJECT_ROOT/requirements.txt" <<'PY'
import sys
from pathlib import Path
from packaging.requirements import Requirement
from packaging.version import Version

path = Path(sys.argv[1])
conflicts = []
for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    try:
        req = Requirement(line)
    except Exception as exc:
        conflicts.append(f"第 {number} 行无法安全解析：{line!r} ({exc})")
        continue
    name = req.name.lower().replace("_", "-")
    if name in {"tensorflow-cpu", "tensorflow-gpu", "tf-nightly"}:
        conflicts.append(f"第 {number} 行包含冲突包：{line}")
    elif name == "tensorflow" and req.specifier and Version("2.21.0") not in req.specifier:
        conflicts.append(f"第 {number} 行排除 TensorFlow 2.21.x：{line}")
if conflicts:
    print("WARNING: requirements.txt 可能降级、替换或破坏 tensorflow[and-cuda]==2.21.*：", file=sys.stderr)
    print("\n".join(f"  - {item}" for item in conflicts), file=sys.stderr)
    sys.exit(1)
PY
then
  pip_install_with_retries -r "$PROJECT_ROOT/requirements.txt"
else
  printf 'WARNING: 已跳过 requirements.txt；请人工审查上述冲突。\n' >&2
fi

stage "关键包"
python -m pip list | grep -Ei "tensorflow|keras|nvidia|numpy|protobuf" || true

stage "安装阶段完成"
printf '本脚本未运行 GPU 验证或项目测试。下一步请执行 scripts/verify_wsl_gpu.sh。\n'
