# 患者级数据划分报告（v2）

- 数据根目录：`data/raw/chest_xray`（仅使用外层目录）
- 随机种子：42
- development pool：原始 `train` + 原始 `val`
- 最终测试集：原始 `test`，保持不变
- 目标比例：train 80%，validation 20%

## 为什么不使用官方 16 张验证集

官方验证集只有 16 张图像（NORMAL 8、PNEUMONIA 8），样本过少，无法稳定估计泛化性能，并且其 50/50 类别比例与开发池的实际类别比例差异明显。因此将原始 train 与 val 合并为 development pool，再按患者分组生成固定的 80/20 划分。原始 test 的 624 张图像不参与此过程，只保留用于最终评估。

## 原始数据结构

本阶段明确读取外层 `data/raw/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}`。原始图像总计 5,856 张：development pool 5,232 张，test 624 张。压缩包中的嵌套 `chest_xray/chest_xray` 和 `__MACOSX` 未删除、未修改，也未纳入清单。

## patient_id 提取规则

- PNEUMONIA：提取文件名开头的 `person<数字>`，正则为 `^(person\d+)_`。例如 `person100_bacteria_475.jpeg` 和 `person100_virus_184.jpeg` 均归为 `person100`。
- NORMAL：完整检查 1,583 个正常文件名后，确认全部匹配 `^(IM-\d+)-` 或 `^(NORMAL2-IM-\d+)-`。尾部检查/视图编号不属于患者 ID，例如 `IM-0011-0001-0002.jpeg` 归为 `IM-0011`。
- 所有 ID 在清单中统一转为小写，以避免大小写造成伪分组。

全部 5,856 张图像均成功解析，共得到 3,118 个 patient_id；无法解析文件为 0。未发现同一 patient_id 同时具有 NORMAL 和 PNEUMONIA 标签。每位患者对应的图像数分布完整保存在 `split_summary.json`，范围为 1–31 张。

## 新划分方法

先按标签和 patient_id 聚合 development pool。对 NORMAL、PNEUMONIA 分别使用固定种子派生的确定性顺序选择验证患者，使各类别验证图像数尽量接近该类别的 20%；同一患者的全部图像一起进入 train 或 val。CSV 按标签、patient_id、文件名稳定排序，因此相同数据与种子会生成相同内容。

## 类别与患者分布

| split | 图像 | 患者 | NORMAL | PNEUMONIA | NORMAL 比例 | PNEUMONIA 比例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 4,185 | 2,299 | 1,079 | 3,106 | 25.7826% | 74.2174% |
| val | 1,047 | 562 | 270 | 777 | 25.7880% | 74.2120% |
| test | 624 | 427 | 234 | 390 | 37.5000% | 62.5000% |

development pool 的实际图像划分为 79.99% / 20.01%，且 train 与 val 的类别比例几乎一致。

## 患者级隔离检查

- train 与 val 的 patient_id 交集：0，通过。
- 所有 development pool 图像均且仅出现于 train 或 val 一次：通过。
- test 保持官方原始 624 张图像不变：通过。

额外检查发现，按上述文件名规则，官方 test 与 train 有 130 个 patient_id 重合，与 val 有 40 个 patient_id 重合。这是原始官方目录结构中的患者编号复用；由于本阶段明确要求 test 完全不变，脚本没有移动或重新分配 test。完整交集列表保存在 `split_summary.json`。

## 哈希重复检查

所有图像计算 SHA-256。train–val、train–test、val–test 三组哈希交集均为空，通过。所有清单引用均真实存在且可由 Pillow 验证打开。

## 无法解析的文件

无。`unparsed_count` 为 0，`unparsed_files` 为空列表。

## 局限性

- patient_id 来自文件名约定，没有外部患者元数据可作独立核验。
- 官方 test 与 development pool 存在 patient_id 编号交叉，虽然没有内容完全相同的图像，但可能造成患者级信息泄漏；最终指标应明确披露这一点。
- patient_id 分组大小不一，因此只能在保持患者完整性的前提下逼近 80/20，而不能保证患者数和图像数同时精确为 80/20。
- 类别严重不平衡；本阶段只保持比例，没有进行采样或训练期加权。
