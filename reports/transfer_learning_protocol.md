# 阶段 5A：迁移学习实验协议

## 参照与候选筛选

CNN baseline 三随机种子的 validation ROC-AUC 为 `0.9743 ± 0.0063`，PR-AUC 为
`0.9897 ± 0.0029`，作为迁移学习候选的参照。VGG16 与 EfficientNetB0 首先仅以 seed 42
运行，用统一协议筛选候选；在确认训练链路和模型行为前，不投入多随机种子成本。

## 两阶段训练

阶段 1 冻结整个 ImageNet backbone，只训练统一分类头，Adam 学习率为 0.001，最多 8
epochs。阶段 2 从 phase1 最低 `val_loss` checkpoint 开始，按模型规则部分解冻，重新
compile，并以 0.00001 学习率最多训练 20 epochs。两个阶段使用全新的 EarlyStopping、
ReduceLROnPlateau、ModelCheckpoint、CSVLogger 和 TerminateOnNaN 实例。最终比较两个阶段
最低 `val_loss`，只将真正最优 checkpoint 复制为 `best_model.keras`。

Pilot 将两个阶段都固定为 1 epoch，只验证权重加载、预处理、冻结/解冻、显存、前反向传播
和产物链路，Pilot 指标不用于模型优劣比较。

## 模型与预处理

两个模型接收 224×224×3、float32、0–255 RGB 图像，使用相同的 GlobalAveragePooling、
Dense(128, ReLU)、Dropout(0.4) 与 sigmoid 分类头，均不使用 Flatten。

- VGG16：模型图内应用官方 `vgg16.preprocess_input`，不使用 `Rescaling(1/255)`；phase2
  仅解冻 `block5_conv1`、`block5_conv2`、`block5_conv3`。
- EfficientNetB0：依赖应用模型内置预处理，不额外归一化；backbone 始终以
  `training=False` 调用。phase2 最多解冻末端 20 个具有权重的非 BatchNormalization 层，
  所有 BatchNormalization 层持续冻结。

冻结 EfficientNet BatchNormalization 并保持 inference mode，可避免小 batch（16）下移动
均值和方差被不稳定地重估。微调采用更低学习率，以降低破坏预训练特征的风险。

## 固定实验约束

batch size 固定为 16，这是 RTX 4060 Laptop GPU 显存与两种候选网络统一比较的硬件约束；
OOM 时立即失败，不自动改变 batch size。继续使用 v3_clean 的既有轻量增强，不增加水平翻转、
MixUp 或 CutMix。

本阶段不使用类别权重、Focal Loss、mixed precision 或阈值优化，以保持与 CNN baseline 的
基本损失和固定阈值协议可比较。验证阈值始终为 0.5。

## 数据隔离

训练只加载 `train.csv`，模型选择和指标只使用 `val.csv`。`test.csv` 仅允许参与 SHA-256
完整性检查，不建立 Dataset、不预测、不计算指标。当前阶段所有结果均是 validation 结果，
不能表述为最终测试性能或临床性能。

## WSL 数据读取稳定性

在 Ubuntu-D 直接读取 `/mnt/d` 项目和数据集时，TensorFlow 对 Windows 挂载盘的并发 JPEG
读取曾触发间歇性 `Input/output error` / `Invalid argument`。为保持不移动数据集、不复制
原始图片的约束，transfer 配置固定使用串行图片读取：

- `data_num_parallel_calls: 1`
- `augmentation_num_parallel_calls: 1`
- `data_prefetch: false`

该调整只影响输入管线并发度，不改变图片、manifest、增强参数、模型结构、损失函数或评估协议。
正式 Full 结果见 `reports/transfer_learning_results.md`。
