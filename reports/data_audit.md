# Chest X-ray 数据审计报告

- 审计日期：2026-07-12（Asia/Shanghai）
- Kaggle 数据集：`paultimothymooney/chest-xray-pneumonia`
- 审计根目录：`<PROJECT_ROOT>\data\raw\chest_xray`
- 图像有效性检查：Pillow `Image.verify()`
- 内容重复检查：SHA-256（仅比较 `train`、`val`、`test` 之间）

## 目录结构

外层 `chest_xray` 目录直接包含 `train`、`val`、`test`，因此选为本次审计根目录。压缩包还解压出了嵌套的 `chest_xray/chest_xray` 副本以及 `__MACOSX` 元数据目录；它们不计入以下六个目标目录的统计。

## 有效图像数量

| 目录 | 有效图像数 |
| --- | ---: |
| `train/NORMAL` | 1,341 |
| `train/PNEUMONIA` | 3,875 |
| `val/NORMAL` | 8 |
| `val/PNEUMONIA` | 8 |
| `test/NORMAL` | 234 |
| `test/PNEUMONIA` | 390 |
| **合计** | **5,856** |

## 检查结果

| 检查项 | 结果 |
| --- | --- |
| 非图像文件 | 未发现（六个目标目录内） |
| 无法打开的损坏图像 | 未发现 |
| 文件名重复 | 未发现（不区分大小写） |
| `train`、`val`、`test` 间内容完全相同的图像 | 未发现（SHA-256） |

## 结论

六个目标目录中的 5,856 个文件均为可验证打开的图像。未发现会影响后续训练、验证或测试拆分完整性的上述数据质量问题。审计过程仅进行读取和哈希计算，没有移动、修改或删除原始图像。
