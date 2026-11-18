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

来源角色划分已落盘为`source80-v1`：只在clean TRAIN内按原始语句ID、
NFKC/casefold/空白归一文本、录音文件名及已有音频SHA的传递闭包分组，
再按固定SHA分桶80/20。得到11448个来源连接组，learning可用40713录音、
selection可用9673录音；另1条冲突录音标记conflict，不参与监督。
独立检查来源ID完整、所有文本/文件/音频hash键跨角色交集为零。
见[来源划分核验](slurp-clean-source-split-v1-verified.json)。7项测试覆盖传递
关系、输入顺序无关、冲突音频传播、缺失/重复/非TRAIN拒绝。

这一划分不是说话人隔离，不覆盖语义近重复；未读取TEST。规则族/表达模板
划分与可训练release仍待完成，不能仅凭source80名单解除训练数据准入限制。

训练适配后续：[逐决策审核适配](slurp-reviewed-training-adapter-v1.md)已接入现有
SFT/RL 训练入口，旧 SLURP 或无审核样本拒绝优化。真实 learning 全量空审核
预检输出 0 条训练样本。此实现不解除语义审核与可训练 release 的质量限制。

后续已单独发布并训练[时间角色工程先导](archive/raw/complex-tasks-deferred/slurp-temporal-pilot-v1.md)：48 个 learning
来源经参考文本语义审核，142 项判断准入、2 项隔离；录音 hash 重新验证但未逐条
听审。物理 4 号卡完成 24 次 SFT 更新。仅限该明确版本，其他 SLURP 数据仍受
原限制；此次没有整体基线、决策链收益或泛化成绩。
