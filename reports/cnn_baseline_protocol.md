# 阶段 4A：CNN baseline 实验协议

## CNN baseline 架构

`cnn_baseline_v1` 接收形状为 224×224×3、像素范围为 0–255 的 RGB 图像。模型内部首先
执行 `Rescaling(1/255)`。四个卷积块依次使用 32、64、128、256 个 3×3 卷积核，每个
卷积后接 Batch Normalization 和 ReLU；前三块包含最大池化。分类头为 Global Average
Pooling、128 单元 ReLU 全连接层、0.4 Dropout 和一个 sigmoid 输出。模型不使用 Flatten、
预训练权重、类别权重、Focal Loss 或 mixed precision。

## 主数据协议

唯一数据配置为 `configs/data_v3_clean.yaml`。模型开发只加载：

- train：`data/splits/v3_clean/train.csv`（3821 张）
- validation：`data/splits/v3_clean/val.csv`（954 张）

正类固定为 PNEUMONIA=1。`test.csv` 可以保留在主数据配置中用于记录最终协议，但阶段 4A
训练入口不会加载它，也不会对测试集预测或计算指标。

## 为什么开发阶段不读取测试集

测试集必须留作最终一次性、无偏的泛化评估。开发阶段查看测试预测或指标会把模型选择、
超参数和研究判断间接拟合到测试集，造成信息泄漏并高估性能。因此本阶段所有 early stopping、
checkpoint 选择和指标报告只依赖训练集与验证集。

## 数据增强

训练集沿用 v3_clean 数据配置中已启用的轻量增强：小角度旋转、平移、缩放和对比度扰动。
验证集不应用增强。输入归一化由模型内部完成。

## 优化器、损失与 callbacks

- Adam，初始学习率 0.001
- BinaryCrossentropy
- ModelCheckpoint：按 `val_loss` 保存唯一的 `best_model.keras`
- EarlyStopping：patience=5、min_delta=0.0001、恢复最优权重
- ReduceLROnPlateau：factor=0.5、patience=2、最低学习率 1e-6
- CSVLogger 与 TerminateOnNaN

Keras 的 binary accuracy、precision、recall、ROC-AUC 和 PR-AUC 仅用于训练监控。最终验证
指标由完整验证集预测一次性使用 sklearn 计算。

## 固定阈值与验证指标

决策阈值固定为 0.5，本阶段不优化阈值。完整验证集报告 accuracy、precision、sensitivity
（recall）、specificity、F1、balanced accuracy、ROC-AUC、PR-AUC、NPV、Brier score 以及
TN、FP、FN、TP。specificity 定义为 TN/(TN+FP)，NPV 定义为 TN/(TN+FN)，所有除零情况
安全返回 0。precision 不会被误写为 specificity。

## Pilot 与 full

- pilot：最多 3 epochs，用于确认正式实验代码、GPU、callbacks 和产物链路正常。
- full：最多 30 epochs，并允许 early stopping；当前仅提供脚本，不在 Codex agent 中运行。

两者写入不同的 `pilot/` 与 `full/` 时间戳目录，任何运行都不会覆盖既有实验。只保存
`val_loss` 最优的 `best_model.keras`，不保存 `final_model.keras`。

## 类别不平衡暂不处理

本 baseline 有意不使用类别权重或 Focal Loss，以建立简单、可解释的未校正参考点。类别
不平衡策略属于后续受控实验变量，应在固定数据划分和评价协议下单独比较，避免同时改变多个
因素而无法归因。

## 局限性

- 单一轻量 CNN 架构，尚未比较迁移学习模型。
- 单一种子不能刻画训练方差；后续需要多种子重复实验。
- 固定阈值 0.5 未针对临床工作点调整。
- 验证集指标不是外部验证或最终测试性能。
- 当前不进行概率校准、亚组分析、置信区间或显著性检验。
- 胸片二分类结果不能直接用于临床诊断。
