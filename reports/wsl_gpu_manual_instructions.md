# WSL2 TensorFlow GPU 手动配置与验证

配置日期：2026-07-12

## 为什么不能用 Codex agent 的 GPU 检查结果判断 WSL GPU 状态

Codex agent 运行在受限的执行沙箱中。沙箱可能阻止 NVML/GPU 设备访问，因此其中运行
`nvidia-smi` 出现 `GPU access blocked by the operating system`，并不代表普通 WSL2
终端中的 GPU 链路异常。GPU 可用性应以用户直接打开的 `Ubuntu-D` 终端中的结果为准。

## 已人工确认的环境

- WSL2 发行版：`Ubuntu-D`
- Windows NVIDIA 驱动：`592.27`
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU
- 显存：8188 MiB
- Driver Model：WDDM
- `/usr/lib/wsl/lib/nvidia-smi` 可正常运行
- `/dev/dxg` 存在
- `/usr/lib/wsl/lib/libcuda.so`、`libcuda.so.1`、`libcuda.so.1.1` 存在

WSL 使用 Windows 主机提供的 GPU 驱动桥接。不得在 WSL 内安装 NVIDIA Linux 显示驱动，
也不得通过 `apt` 安装 `cuda-toolkit`、`cudnn` 或 `nvidia-driver`。

## 为什么使用 Python 3.11

Ubuntu-D 当前系统 Python 是 3.14.4，而 TensorFlow 2.21 不支持 Python 3.14。
脚本使用 `uv` 安装独立的 Python 3.11，并将虚拟环境放在
`~/.venvs/chest-xray`，与 Windows Python 环境完全分离。

## 脚本用途

- `scripts/setup_wsl_gpu.sh`：检查 WSL GPU 基础链路，安装/准备 `uv` 和 Python 3.11，
  创建独立虚拟环境并安装 TensorFlow 2.21 GPU 依赖及项目依赖。它不运行 GPU 验证或测试。
- `scripts/verify_wsl_gpu.sh`：验证 Python/TensorFlow 版本、CUDA 构建状态、可见 GPU，
  并在 `/GPU:0` 上执行 2048×2048 矩阵乘法。失败时不会安装系统 CUDA 包或创建链接。
- `scripts/run_wsl_regression.sh`：先运行 GPU 验证，再核验数据管道遍历数量、pytest、
  Smoke Test，并比较四份 `v3_clean` manifest 运行前后的 SHA-256。

Smoke Test 仅执行代码中限定的两个训练 step，不保存模型权重，也不是正式训练。

TensorFlow 的 CUDA/cuDNN 运行库由 pip 安装在虚拟环境的 `site-packages/nvidia/*/lib`
目录中。验证和回归脚本仅在自身进程及其子进程内临时设置 `LD_LIBRARY_PATH`，让动态加载器
找到这些库；不会修改系统配置、安装 apt CUDA 包或创建符号链接。脚本退出后该设置不会
影响启动它的终端。

`v3_clean` manifest 保留了创建时的 Windows 绝对路径（例如 `<WINDOWS_ABSOLUTE_PATH>`）。数据管道在
Windows 中继续按原方式解析；在 WSL/POSIX 中会在内存里将盘符路径映射到常规挂载点
（例如 `<WSL_MOUNT_PATH>`）。此兼容层不会重写或修改 manifest。

## 手动执行命令

请在 Windows PowerShell 中明确进入正确的发行版：

```powershell
wsl -d Ubuntu-D
```

随后在普通 Ubuntu-D 终端中按顺序运行：

```bash
cd "<PROJECT_ROOT>"
chmod +x scripts/setup_wsl_gpu.sh scripts/verify_wsl_gpu.sh scripts/run_wsl_regression.sh
./scripts/setup_wsl_gpu.sh && \
./scripts/verify_wsl_gpu.sh && \
./scripts/run_wsl_regression.sh
```

使用 `&&` 可确保前一阶段失败时不执行下一阶段。三个阶段应逐个确认成功；任一阶段失败时
停止，并保留完整终端输出用于诊断。TensorFlow wheel 较大；安装脚本已启用延长超时、更多
连接重试、断点续传和命令级重试。项目依赖会逐包安装，因此网络中断时只重试当前包。
若仍因网络中断失败，可直接重新运行安装脚本，它会复用已有环境和已安装的软件包。

## Git 注意事项

实际虚拟环境位于项目目录外的 `~/.venvs/chest-xray`。不得把虚拟环境提交到 GitHub。
项目 `.gitignore` 也保留了 `.venv/`、`.venv-wsl/`、Python 缓存和 pytest 缓存规则，
以防将来在项目目录内误建环境或缓存。
