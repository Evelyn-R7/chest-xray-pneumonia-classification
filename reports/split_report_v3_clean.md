# 患者完全隔离数据划分报告（v3_clean）

- 协议：`patient_isolated_official_test`
- 数据根目录：`data/raw/chest_xray`
- 随机种子：42
- development pool：原始 train + 原始 val
- test：原始官方 test 624 张，清单与 v2 byte-for-byte 一致

## 为什么哈希不重复仍可能存在患者级泄漏

SHA-256 只能识别文件内容完全相同的图像。同一患者可以在不同时间、不同投照位置或不同处理版本下产生多张内容不同的 X 光片；这些文件的 SHA-256 不同，但仍共享患者特有的解剖结构、设备或采集特征。因此，仅有零哈希交集不能证明患者级独立。v3_clean 同时要求 train、val、test 的 patient_id 两两无交集。

## patient_id 的证据边界

patient_id 来自文件名约定，而非外部患者主索引：PNEUMONIA 使用 `^(person\d+)_`；NORMAL 使用 `^(IM-\d+)-` 或 `^(NORMAL2-IM-\d+)-`。规则覆盖全部图像，但没有医院级元数据可独立验证编号是否始终等价于真实患者身份。

## 为什么保留 v2

v2 完整遵循官方 test benchmark，可与使用同一公开测试集的既有研究直接比较，因此保留其清单和报告且不作修改。不过，v2 的官方 test 与 development pool 存在 170 个按文件名推断的 patient_id 重合，可能使最终指标偏乐观。

## 为什么 v3_clean 是主实验协议

v3_clean 保持官方 test 的 624 张图像完全不变，同时从 development pool 排除所有与 test patient_id 重合的患者，再对剩余患者重新生成 train/val。这样兼顾官方测试集可比性与 train、val、test 三方患者级隔离，适合作为主实验和模型选择协议；v2 仅作为 official-test benchmark 补充报告。

## 排除统计与类别变化

原始 development pool 有 5,232 张图像、2,861 位患者：NORMAL 1,349（25.7836%），PNEUMONIA 3,883（74.2164%）。

因与 test patient_id 重合，共排除 170 位患者、457 张图像：NORMAL 0，PNEUMONIA 457。清理后的 development pool 有 4,775 张图像、2,691 位患者：NORMAL 1,349（28.2513%），PNEUMONIA 3,426（71.7487%）。排除项全部来自 PNEUMONIA，因此正常类比例提高约 2.47 个百分点。

## v3_clean 最终统计

| split | 图像 | 患者 | NORMAL | PNEUMONIA | NORMAL 比例 | PNEUMONIA 比例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 3,821 | 2,146 | 1,079 | 2,742 | 28.2387% | 71.7613% |
| val | 954 | 545 | 270 | 684 | 28.3019% | 71.6981% |
| test | 624 | 427 | 234 | 390 | 37.5000% | 62.5000% |

清理后 development pool 的图像比例为 train 80.02%、val 19.98%。分标签进行患者级分组后，train 与 val 的类别比例基本一致。

## 严格验证结果

- train–val patient_id 交集：0，通过。
- train–test patient_id 交集：0，通过。
- val–test patient_id 交集：0，通过。
- train–val SHA-256 交集：0，通过。
- train–test SHA-256 交集：0，通过。
- val–test SHA-256 交集：0，通过。
- v3 test 与 v2 test 文件 SHA-256：均为 `6ba1a4093a601c590b433e99e8a332b2af3a5f46c29f062dd8e1d283f3a2622d`，完全一致。
- 457 张排除图像均未进入 v3 train 或 val，通过。
- 剩余 development pool 每张图像恰好出现一次，通过。
- 所有清单与排除清单引用的图像均存在且可由 Pillow 打开，通过。

## 局限性

- 文件名推断的 patient_id 缺少外部临床元数据验证，可能存在编号碰撞或编号语义不一致。
- 为保持官方 test 不变，v3_clean 必须舍弃 457 张开发图像，降低了可用于训练和验证的数据量。
- 排除项全部为 PNEUMONIA，改变了 development pool 的类别先验；训练和指标解释应考虑此变化。
- 患者组大小不同，划分只能逼近 80/20，无法同时精确匹配图像数、患者数和类别数。
- 患者隔离不能排除相同设备、机构或采集流程造成的其他域泄漏。
