# SLURP 标注质量约束：2026-11-11

用户明确要求不拿 omni 已发现的 SLURP 标注缺陷对应指标进行训练。
在本项目完成清单对齐和数据验收前，SLURP 指标只作诊断；不进入 SFT/RL
监督、奖励、教师优化反馈、候选筛选或模型选择。既有冻结 TEST 和仅验证输入
接口的 TRAIN 推理保留，不能把名称中的 TRAIN 当作已经发生权重训练。
此文件是当前执行约束，不代表所有历史程序已经实现自动拒绝。

已读取 omni 发布目录的 clean-v7 report，而非仍写 clean-v5 的旧概览页：
`/home/xingchao/workspace/omni/repro/publication/omni/docs/results/dataset-quality`。
公开地址：https://github.com/chaosxingxc-orion/omni/tree/main/docs/results/dataset-quality 。

| 划分 | 排除文本 / 录音引用 | 保留文本 / 录音引用 |
|---|---:|---:|
| train | 54 / 240 | 11460 / 50388 |
| test | 27 / 87 | 2947 / 12991 |
| train_synthetic | 59 / 151 | 19652 / 69102 |
| devel | 0 / 0 | 2033 / 8690 |

文件 SHA256：

- report.json：`901826f04f402ae765f0400daecd2c5e27ced9350985d0e95d9e4852f9b7ded8`
- exclusions.jsonl：`a3cd74482c32b74f117fb61489eeb3466a37f30f18aa304fc59c2260ab5649d3`
- apply_release.py：`1e5f6bd013217a99b6ec0f5844aa784e66355540c24cb9bad888260c643ec42b`

确认问题包括意图与句义冲突、槽位类型错标、输入缺任务信息和字段跨业务错配。
这是已确认排除清单，不证明剩余数据全量干净；未听审音频，不称音频损坏。
测试原句、gold和缺陷案例不得进入教师或训练。处理清单只按固定身份与哈希
匹配，不通过模型错误、低分、标签频率或相似度扩展排除。

下一步：认证本项目原 metadata 与发布源 SHA，按清单建立新版本 metadata，
将录音及所有 Jev 派生决策同步映射；分别验收 learning/selection/TEST。
旧运行保留原分母。后续如重算清洗后 TEST，注明这是人工质量审查后的派生
评测，披露排除数，不作为训练反馈，也不与旧分母直接计算算法提升。

omni 官方审计另发现 synthetic 与 devel/TEST 存在来源及同文重叠。因此不能
将完整 synthetic 合入训练后宣称独立泛化，需另做来源与同文隔离。
原始数据及 gold 不覆盖、不删除；当前没有 SLURP 权重训练派发。

2026-11-11 后续对齐已完成：四份原始metadata SHA与clean-v7发布一致，
新metadata在`/mnt/cephfs/xingchao/datasets/slurp-clean/clean-v7-a3cd74482c32b74f`
独立重建，逐行与原始记录减固定排除清单的结果一致。
见[metadata及原生请求对齐](slurp-clean-v7-alignment-v1.json)和
[Jev整录音排除映射](slurp-clean-v7-jev-exclusion-map-v1.json)。
TRAIN排除240个bundle/40320决策，TEST排除87个bundle/14616决策。
这只是映射，尚未发布可训练的clean Jev版本或验收learning/selection。
TRAIN剩余50388条录音引用对应50387个文件，既有重复文件的冲突标签仍存在。
因此质量限制没有解除，也没有产生清洗后的性能分数。
