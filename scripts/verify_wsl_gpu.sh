#!/usr/bin/env bash
set -euo pipefail

VENV="$HOME/.venvs/chest-xray"
stage() { printf '\n===== %s =====\n' "$1"; }
fail() { printf 'ERROR: %s\n' "$1" >&2; exit 1; }
configure_venv_cuda_paths() {
  local nvidia_root="$VENV/lib/python3.11/site-packages/nvidia"
  local nvidia_libs
  [[ -d "$nvidia_root" ]] || fail "虚拟环境 NVIDIA 包目录不存在：$nvidia_root"
  nvidia_libs="$(find "$nvidia_root" -mindepth 2 -maxdepth 2 -type d -name lib -print | sort | paste -sd: -)"
  [[ -n "$nvidia_libs" ]] || fail "未找到虚拟环境内的 NVIDIA 动态库目录"
  export LD_LIBRARY_PATH="$nvidia_libs:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  printf '已为当前脚本临时配置虚拟环境 CUDA 动态库搜索路径。\n'
}

stage "激活 Python 3.11 虚拟环境"
[[ -f "$VENV/bin/activate" ]] || fail "虚拟环境不存在：$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -c 'import tensorflow' >/dev/null 2>&1 \
  || fail "TensorFlow 尚未安装完成；请先重新运行 scripts/setup_wsl_gpu.sh，成功后再验证"
configure_venv_cuda_paths

stage "检查 WSL GPU 链路"
[[ -x /usr/lib/wsl/lib/nvidia-smi ]] || fail "/usr/lib/wsl/lib/nvidia-smi 不存在或不可执行"
[[ -e /dev/dxg ]] || fail "/dev/dxg 不存在"
[[ -f /usr/lib/wsl/lib/libcuda.so ]] || fail "/usr/lib/wsl/lib/libcuda.so 不存在"
/usr/lib/wsl/lib/nvidia-smi

stage "验证 TensorFlow GPU"
python - <<'PY'
import sys

if sys.version_info[:2] != (3, 11):
    raise RuntimeError(f"Python 必须为 3.11，当前为 {sys.version}")

try:
    import tensorflow as tf
except Exception:
    print("TensorFlow 导入失败。不会自动安装 apt CUDA 包或创建符号链接。", file=sys.stderr)
    raise

print("Python:", sys.version)
print("TensorFlow:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("Build info:", tf.sysconfig.get_build_info())

if not tf.__version__.startswith("2.21."):
    raise RuntimeError(f"TensorFlow 必须为 2.21.x，当前为 {tf.__version__}")
if not tf.test.is_built_with_cuda():
    raise RuntimeError("TensorFlow 未构建 CUDA 支持")

gpus = tf.config.list_physical_devices("GPU")
print("Visible GPU devices:", gpus)
print("GPU count:", len(gpus))
if not gpus:
    print("诊断：TensorFlow 导入成功但 GPU 列表为空。", file=sys.stderr)
    print("请人工检查 pip 中的 tensorflow/nvidia 包及 TensorFlow build info；", file=sys.stderr)
    print("本脚本不会安装 apt CUDA 包，也不会创建任何符号链接。", file=sys.stderr)
    raise RuntimeError("TensorFlow 未检测到 GPU")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

with tf.device("/GPU:0"):
    a = tf.random.normal((2048, 2048))
    b = tf.random.normal((2048, 2048))
    c = tf.matmul(a, b)

print("Matrix result device:", c.device)
if "GPU" not in c.device.upper():
    raise RuntimeError(f"矩阵计算没有运行在 GPU 上：{c.device}")
print("GPU verification: SUCCESS")
PY
