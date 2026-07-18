# 阶段 4A：CNN baseline 运行结果

验证日期：2026-07-14 至 2026-07-15

## 运行状态

CNN baseline pilot 与 full 均在普通 Ubuntu-D 终端中成功完成。TensorFlow 2.21.0 检测到
NVIDIA GeForce RTX 4060 Laptop GPU，CUDA、cuDNN 和 GPU 矩阵前置验证通过。模型参数量
为 422,881。

本报告只包含 `v3_clean` 验证集结果。训练代码没有加载 `test.csv`，没有生成测试集预测，
也没有计算测试集指标。

## Pilot

- 输出目录：`results/experiments/cnn_baseline_v1/seed_42/pilot/20260714T150718_662420Z`
- 运行 epochs：3
- 最佳 epoch：2（`val_loss=2.45560`）
- accuracy：0.716981
- precision：0.716981
- sensitivity / recall：1.000000
- specificity：0.000000
- F1：0.835165
- balanced accuracy：0.500000
- ROC-AUC：0.768665
- PR-AUC：0.895580
- NPV：0.000000
- Brier score：0.282908
- confusion matrix：TN=0、FP=270、FN=0、TP=684

Pilot 在固定阈值 0.5 下把全部验证样本预测为阳性。其用途是验证训练和产物链路，不能作为
最终 baseline 性能结论。ROC-AUC 高于 0.5 表明概率排序已有一定信息，但早期训练的概率
尺度尚未形成有效的固定阈值分类。

## Full

- 输出目录：`results/experiments/cnn_baseline_v1/seed_42/full/20260714T153220_856665Z`
- 最大 epochs：30
- 实际运行 epochs：12
- EarlyStopping：在 epoch 12 触发
- 最佳 epoch：7（`val_loss=0.23602`）
- accuracy：0.907757
- precision：0.986928
- sensitivity / recall：0.883041
- specificity：0.970370
- F1：0.932099
- balanced accuracy：0.926706
- ROC-AUC：0.981319
- PR-AUC：0.993055
- NPV：0.766082
- Brier score：0.067923
- confusion matrix：TN=262、FP=8、FN=80、TP=604

Full 运行通过 ModelCheckpoint 保留 epoch 7 的最低 `val_loss` 模型，并在完整验证集上以固定
阈值 0.5 一次性计算上述指标。后续 epoch 的验证损失波动较大，说明单次训练仍存在不稳定性；
后续结论应通过预先规定的多种子重复实验确认，而不能根据本次验证结果继续调参并报告为无偏
性能。

## 产物核验

Pilot 和 full 各自的运行目录均包含且非空：

- `resolved_config.yaml`
- `environment.json`
- `model_summary.txt`
- `history.csv`
- `best_model.keras`
- `val_predictions.csv`
- `val_metrics.json`
- `run_summary.md`
- `learning_curves.png`
- `confusion_matrix_val.png`
- `roc_curve_val.png`
- `pr_curve_val.png`

Full 的 `val_predictions.csv` 包含 954 条验证预测，字段符合协议要求。没有创建
`final_model.keras`。`results/` 与 `*.keras` 已被 `.gitignore` 排除，不应提交到 Git。

## 数据完整性

Pilot 和 full 运行前后，四份 `v3_clean` manifest 的 SHA-256 均保持不变：

- train：`fac67671d85f11d66bfa87179fb1027cb51da59fa163936102282da31352f566`
- val：`ab7dcd0425ec69e379f248ffe1a4875d5f50534c75ff9d8860fd5a7ec3f696f0`
- test：`6ba1a4093a601c590b433e99e8a332b2af3a5f46c29f062dd8e1d283f3a2622d`
- excluded：`c9a8284ced63608ceeea95f757d7c8a3d2fdb77bf2e633bacae02d7aa36bd522`

原始图片和 manifest 均未修改。测试 manifest 仅参与运行前后的文件哈希完整性校验，没有被
训练或验证数据管道加载。

## 当前结论

阶段 4A 的正式实验框架、CNN baseline pilot 和 seed 42 full run 已完成。Full 验证结果表明
该 CNN 能在当前验证划分上形成较强的区分能力，但这仍是单种子内部验证结果，不代表最终
测试集性能或临床有效性。此处停止，不读取测试集，也不根据测试集进行任何模型选择。
