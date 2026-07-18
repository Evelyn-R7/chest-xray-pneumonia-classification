# WSL2 TensorFlow GPU 环境与阶段 3 回归报告

## 环境概况

- 配置与验证日期：2026-07-12 至 2026-07-14
- WSL 发行版：Ubuntu-D（WSL2）
- WSL BasePath：`<LOCAL_WSL_BASEPATH>`
- Ubuntu：26.04 LTS（Resolute Raccoon）
- WSL 内核：`6.18.33.2-microsoft-standard-WSL2`
- Python：3.11.15（uv 管理，虚拟环境 `~/.venvs/chest-xray`）
- TensorFlow：2.21.0
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU
- GPU 显存：8188 MiB
- Windows NVIDIA 驱动：592.27
- nvidia-smi CUDA Version：13.1
- Driver Model：WDDM
- WSL 根文件系统剩余空间：943 GB
- Windows D 盘及 `/mnt/d` 剩余空间：206 GB

## TensorFlow GPU 验证

- `tf.test.is_built_with_cuda()`：True
- TensorFlow CUDA build：12.5.1
- TensorFlow cuDNN build：9
- CUDA compute capabilities：sm_60、sm_70、sm_80、sm_89、compute_90
- TensorFlow 检测到的 GPU：1 个，`/physical_device:GPU:0`
- 矩阵计算实际设备：`/job:localhost/replica:0/task:0/device:GPU:0`
- GPU 验证结果：SUCCESS
- TensorFlow 为 GPU 进程分配的可用显存：5560 MB

Smoke Test 运行时记录：

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，Compute Capability 8.9
- Driver：13.1.0
- Runtime：12.9.0
- Toolkit：12.5.0
- cuDNN：9.24.0
- XLA CUDA 初始化成功，cuDNN 92400 加载成功

CUDA 与 cuDNN 运行库来自 `tensorflow[and-cuda]` 安装到虚拟环境的 pip 包。未在 WSL
中安装 NVIDIA Linux 显示驱动，未通过 apt 安装 CUDA Toolkit、cuDNN 或 nvidia-driver。

## 阶段 3 回归结果

### 数据管道

- train 遍历数量：3821
- val 遍历数量：954
- test 遍历数量：624
- train/val/test 元数据数量：3821 / 954 / 624
- 首个训练 batch：图像 `[32, 224, 224, 3]`，标签 `[32]`
- 最后 batch：train 13、val 26、test 16
- 验证集确定性：通过
- 测试集确定性：通过
- 数据增强改变图像且保持形状和标签：通过
- TensorFlow 数据管道检测到 GPU：`/physical_device:GPU:0`

### pytest

- 结果：15 passed
- 耗时：2599.07 秒（43 分 19 秒）
- 要求：至少保持原有 14 项通过
- 结论：通过

### Smoke Test

- 状态：success
- `steps_per_epoch`：2
- `validation_steps`：1
- GPU/cuDNN/XLA：成功启用
- 未保存模型权重
- 本次仅为有限步数 Smoke Test，不是正式模型训练

Smoke Test 输出的 accuracy/loss 仅用于确认计算链路可运行，`formal_metric=False`，不得视为
正式实验指标。

## manifest SHA-256 前后对比

| manifest | 运行前 SHA-256 | 运行后 SHA-256 | 结果 |
|---|---|---|---|
| `train.csv` | `fac67671d85f11d66bfa87179fb1027cb51da59fa163936102282da31352f566` | `fac67671d85f11d66bfa87179fb1027cb51da59fa163936102282da31352f566` | 未改变 |
| `val.csv` | `ab7dcd0425ec69e379f248ffe1a4875d5f50534c75ff9d8860fd5a7ec3f696f0` | `ab7dcd0425ec69e379f248ffe1a4875d5f50534c75ff9d8860fd5a7ec3f696f0` | 未改变 |
| `test.csv` | `6ba1a4093a601c590b433e99e8a332b2af3a5f46c29f062dd8e1d283f3a2622d` | `6ba1a4093a601c590b433e99e8a332b2af3a5f46c29f062dd8e1d283f3a2622d` | 未改变 |
| `excluded_from_development.csv` | `c9a8284ced63608ceeea95f757d7c8a3d2fdb77bf2e633bacae02d7aa36bd522` | `c9a8284ced63608ceeea95f757d7c8a3d2fdb77bf2e633bacae02d7aa36bd522` | 未改变 |

结论：四份 `v3_clean` manifest 均保持不变。

## 安装和验证过程中遇到的问题

1. Codex 受限 agent 沙箱阻止 NVML/GPU 访问，曾输出 `GPU access blocked by the
   operating system`。普通 Ubuntu-D 终端中的 `/usr/lib/wsl/lib/nvidia-smi` 正常，因此
   GPU 验证均由用户在普通 Ubuntu-D 终端手动执行。
2. Ubuntu 系统 Python 3.14.4 不受 TensorFlow 2.21 支持。使用 uv 安装 Python 3.11.15，
   并创建独立虚拟环境 `~/.venvs/chest-xray`，未触及 Windows Python 环境。
3. TensorFlow 和部分依赖下载期间出现超时、SSL EOF 和 IncompleteRead。安装脚本加入延长
   超时、断点续传、命令级重试和逐包安装后完成安装。
4. pip 安装的 CUDA/cuDNN 动态库位于虚拟环境 `site-packages/nvidia/*/lib`，默认动态库
   搜索路径无法发现。验证与回归脚本仅在自身进程中临时设置 `LD_LIBRARY_PATH`；未安装
   apt CUDA 包，未创建系统或虚拟环境符号链接。
5. manifest 保留 Windows `<WINDOWS_ABSOLUTE_PATH>` 绝对路径。数据管道增加 WSL 路径兼容解析，在内存中
   映射为 `<WSL_MOUNT_PATH>`，没有重写 manifest。
6. 数据管道结束时出现 TensorFlow rendezvous cancellation 日志，但程序正常输出完整验证
   JSON、数量核验通过且退出成功，不影响本次回归结论。

## 实际执行的关键命令

```bash
cd "<PROJECT_ROOT>"
./scripts/setup_wsl_gpu.sh
./scripts/verify_wsl_gpu.sh
./scripts/run_wsl_regression.sh
```

回归脚本内部依次执行：

```bash
python src/check_data_pipeline.py
python -m pytest -q
python src/smoke_test.py
```

并在运行前后对以下文件执行 `sha256sum`：

```text
data/splits/v3_clean/train.csv
data/splits/v3_clean/val.csv
data/splits/v3_clean/test.csv
data/splits/v3_clean/excluded_from_development.csv
```

## 最终结论

WSL2 TensorFlow GPU 环境配置成功，RTX 4060 Laptop GPU 可被 TensorFlow 2.21 正确识别
并执行矩阵计算。阶段 3 数据管道、15 项 pytest 与有限步数 Smoke Test 全部通过，四份
manifest 完全未改变。验证完成后停止，未开始正式模型训练。
