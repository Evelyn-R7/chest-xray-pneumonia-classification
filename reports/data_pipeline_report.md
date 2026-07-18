# 阶段 3：数据加载管道与 Smoke Test 报告

## 运行环境

- Python：3.11.4
- TensorFlow：2.21.0
- pytest：9.1.1
- GPU：未检测到；TensorFlow 2.11 及以上在原生 Windows 不提供 CUDA GPU 支持，本次使用 CPU。

## 数据协议与 manifest

- 配置：`configs/data_v3_clean.yaml`
- train：`data/splits/v3_clean/train.csv`
- validation：`data/splits/v3_clean/val.csv`
- test：`data/splits/v3_clean/test.csv`
- 协议：`patient_isolated_official_test`

标签完全读取自 CSV 的 `label` 字段，不依赖目录名称推断。映射固定为 `NORMAL = 0`、`PNEUMONIA = 1`，输出标签类型为 `tf.float32`。

## 数据量与类别

| split | 图像 | NORMAL | PNEUMONIA |
| --- | ---: | ---: | ---: |
| train | 3,821 | 1,079 | 2,742 |
| val | 954 | 270 | 684 |
| test | 624 | 234 | 390 |

完整遍历的实际样本数分别为 3,821、954、624，与 manifest 完全一致。最后一个 batch 分别为 13、26、16，确认 `drop_remainder=False` 且无遗漏。

## 图像与 batch

- 首个训练 batch 图像 shape：`(32, 224, 224, 3)`
- 标签 shape：`(32,)`
- 图像 dtype：`tf.float32`
- 标签 dtype：`tf.float32`
- 实测首批像素范围：`0.0–255.0`
- 通道：RGB 三通道
- 数据管道未统一除以 255，模型内自行预处理。
- 未发现 NaN 或 Inf。
- 默认不 cache 完整解码图像；启用 `prefetch=tf.data.AUTOTUNE`。

## 数据增强

增强仅动态应用于训练集：旋转 0.03、平移高/宽 0.05、缩放高/宽 0.10、对比度 0.10，固定种子 42。未使用水平翻转、剪切、CutMix 或 MixUp，也未保存增强图像。

同一原图连续通过增强层得到不同张量，尺寸始终为 `(224, 224, 3)`，标签不变且无 NaN/Inf。无增强解码时，同一图像重复读取完全一致。验证集和测试集不应用增强。

## 确定性检查

验证集和测试集均连续完整读取两次，对每个 batch 的文件顺序对应标签以及图像张量字节哈希进行比较：两次结果完全一致。训练集使用 seed 42、`reshuffle_each_iteration=True`，每轮动态洗牌。

## pytest

执行 `py -m pytest -q`：14 项测试全部通过，耗时 22.70 秒。测试覆盖标签映射、manifest 行数、文件存在性、shape/dtype、标签值域、验证/测试确定性、增强 shape/标签、非法标签和缺失路径错误。

出现 2 条第三方 `asttokens/astroid` 弃用警告，与本项目数据管道无关。

## Smoke Test

极小 CNN 按要求运行 1 epoch、2 个训练 step、1 个验证 step，前向传播、反向传播、binary crossentropy 和 validation 均成功，所有返回值有限。未保存模型或权重，输出不作为正式实验指标，运行后调用 `tf.keras.backend.clear_session()`。

Keras 提示 `fit` 的默认 shuffle 参数对 `tf.data.Dataset` 不生效；这是预期行为，训练数据已在 tf.data 管道中完成带固定种子的 shuffle。session 清理另触发 TensorFlow 内部 deprecated API 提示，不影响成功结果。

## 可视化检查

- `reports/data_pipeline/train_batch_original.png`
- `reports/data_pipeline/train_batch_augmented.png`
- `reports/data_pipeline/val_batch.png`
- `reports/data_pipeline/test_batch.png`

四张图均显示 16 张图像，只标注类别、patient_id 和文件名，不含完整本地路径。目视检查显示方向正常、标签与 manifest 一致，增强幅度合理。

## 发现的问题及修复

1. 初始环境缺少 TensorFlow 和 pytest。首次安装的 351 MB TensorFlow 下载进程失去网络连接并卡住；重新运行后成功安装 TensorFlow 2.21.0 与 pytest 9.1.1。
2. pip 报告全局环境中的部分 Google/OR-Tools 包与 TensorFlow 2.21.0 所需 protobuf 7.35.1 存在版本约束冲突；TensorFlow 导入、全部管道检查、pytest 和 Smoke Test 均实际成功。本项目 requirements 不固定版本，后续建议用独立虚拟环境隔离其他项目依赖。
3. 完整确定性检查最初若保留两轮全部图像会占用大量内存，已改为逐 batch 计算张量 SHA-256 签名并比较，保持严格检查同时控制内存。

本阶段未进行正式训练、未保存模型权重、未修改原始图像，也未修改 v2/v3_clean manifest。
