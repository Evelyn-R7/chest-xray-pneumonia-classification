# 阶段 5A：迁移学习验证结果

## 运行范围

本阶段在 WSL2 Ubuntu-D、Python 3.11、TensorFlow 2.21.0 GPU 环境下完成 VGG16 与
EfficientNetB0 的 seed 42 两阶段迁移学习 Full 验证。训练只使用 `train.csv`，模型选择与
指标只使用 `val.csv`。`test.csv` 未被加载、预测或评估，仅参与 SHA-256 完整性检查。

由于项目和数据集继续保留在 `/mnt/d`，为避免 WSL Windows 挂载盘上的 TensorFlow 并发读图
错误，Full 运行使用串行图片读取与关闭 prefetch。该设置不改变数据内容、manifest、模型或
评估指标定义。

## 完整性检查

四份 v3_clean manifest 在运行前后 SHA-256 保持一致：

| manifest | SHA-256 |
| --- | --- |
| `data/splits/v3_clean/train.csv` | `fac67671d85f11d66bfa87179fb1027cb51da59fa163936102282da31352f566` |
| `data/splits/v3_clean/val.csv` | `ab7dcd0425ec69e379f248ffe1a4875d5f50534c75ff9d8860fd5a7ec3f696f0` |
| `data/splits/v3_clean/test.csv` | `6ba1a4093a601c590b433e99e8a332b2af3a5f46c29f062dd8e1d283f3a2622d` |
| `data/splits/v3_clean/excluded_from_development.csv` | `c9a8284ced63608ceeea95f757d7c8a3d2fdb77bf2e633bacae02d7aa36bd522` |

## Full 运行目录

| 模型 | 运行目录 | 状态 |
| --- | --- | --- |
| VGG16 | `results/experiments/vgg16_transfer_v1/seed_42/full_20260716T160438_462725Z` | 完整通过 |
| EfficientNetB0 | `results/experiments/efficientnetb0_transfer_v1/seed_42/full_20260716T170609_614096Z` | 完整通过 |

两个目录均包含 `best_model.keras`、`phase1_best.keras`、`phase2_best.keras`、
`combined_history.csv`、`val_predictions.csv`、`val_metrics.json`、`run_summary.md`
以及 validation ROC、PR、confusion matrix 和 learning curve 图。两份 `val_predictions.csv`
均为 955 行，即 954 条验证集预测加表头。

## Validation 指标

| 指标 | VGG16 Full | EfficientNetB0 Full |
| --- | ---: | ---: |
| best phase | phase2 | phase2 |
| phase1 min val_loss | 0.118950 | 0.115543 |
| phase2 min val_loss | 0.092194 | 0.073880 |
| accuracy | 0.961216 | 0.970650 |
| precision | 0.995406 | 0.983776 |
| sensitivity / recall | 0.950292 | 0.975146 |
| specificity | 0.988889 | 0.959259 |
| F1 | 0.972326 | 0.979442 |
| balanced accuracy | 0.969591 | 0.967203 |
| ROC AUC | 0.995869 | 0.996166 |
| PR AUC | 0.998444 | 0.998530 |
| NPV | 0.887043 | 0.938406 |
| Brier score | 0.027044 | 0.022030 |
| TN | 267 | 259 |
| FP | 3 | 11 |
| FN | 34 | 17 |
| TP | 650 | 667 |

## 结论

两个 transfer 候选在 validation 上均超过 CNN baseline seed 42 与 multiseed 均值。按当前固定
阈值 0.5 的 validation 结果，EfficientNetB0 取得更高的 accuracy、sensitivity、F1、ROC AUC、
PR AUC、NPV 和更低的 Brier score；VGG16 取得更高的 precision 与 specificity，误报更少。

这些结果仍属于 validation 筛选结果，不应表述为 test performance 或临床性能。下一阶段如果继续
推进，建议优先将 EfficientNetB0 作为 transfer 候选进入多随机种子验证，同时保留 VGG16 作为高
specificity 对照。
