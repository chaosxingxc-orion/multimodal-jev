# 简单语音任务基础配置

本页登记简单任务的实验范围和公共设置。每次派发以台账研究 ID 对应的冻结协议为准。

| 设置 | 约定 |
| --- | --- |
| 研究范围 | 单音频意图判决、训练效率、同任务跨数据集迁移。多音频、多意图组合、多步判决、SFC 和 Audio2Tool 暂缓。 |
| 数据 | 全量 clean SLURP TRAIN 是源数据。selection 用于选择，SLURP clean TEST 和 MINDS native14 用于独立评测。FSC native31 留在历史对照。 |
| 补充评测 | 范围固定为 [SIB-FLEURS 英语话题和 MMSU 多义词](../docs/research/speech-semantic-transfer-targets.md)。WolBanking77 只保留已审核的历史诊断，YUE-PUB-Speech 不进入派发范围。 |
| 评测 | 单标签报告 Top-1 macro-F1、准确率及平均 Top-1 原始概率。Harness 阈值仅在源域 TRAIN 校准并冻结。固定阈值集合指标用于诊断。 |
| 资源 | 实验在 181 接续，69 的记录用于历史追溯。每次派发重新核对可用设备、镜像、路径、锁和内容身份。 |
| 运行观察 | 普通远程状态检查间隔 900 秒。用户询问或收到终态通知时可即时核验。已冻结运行的监督器继续使用原配置。 |
| 验收 | 短工程窗口按 4 次推理、2 次训练更新登记。单独获准的效率对照每配置至多 100 步。正式训练效率从固定窗口和计时日志采样。 |
| 效率比较 | 对齐策略、检查点、评测人口和输出范围，统计准备到独立评分审核的整组耗时。记录冷启动、缓存和并行重叠。 |

## 两阶段对照

参考模板包括 [Final60 数据与模型](../configs/experiments/slurp-simple-final-v1.json)及[早期 DP6 全量配置](../configs/experiments/context-replica-training-dp6-formal-v1.json)。下文难例与回放对照的实际配置以对应运行包为准。

0.6B 八卡方案采用 LoRA r64/alpha128。Stage1 使用 DP8、全局 batch 512，在全量 clean TRAIN 上以 InfoNCE、显式概率间隔和双侧阈值约束学习。Stage2 使用 DP6、两个 TP1 推理实例、全局 batch 384，通过排序与存在性双信用 GRPO 学习。Stage2 从同轮 Stage1 端点初始化，使用该端点打分构建冻结难例与简单回放视图。固定 reference 全量提前生成，按身份校验后只读消费。

DP6 按实际视图人口 `N` 推导两轮共 `2 × ceil(N/384)` 步和尾批。旧 DP2 包与收据保留原身份。新的设备拓扑独立冻结和验收。阶段间 adapter 的复用或叠加由显式配置控制。[阶段 LoRA 协议](../docs/engineering/stage-lora-policy-v1.md) · [已审核的两阶段对照](../wiki/reports/JEV-R64-S2.md)

采用稳定前缀布局的包将任务规约与完整候选定义放在变动的历史和音频前，决策标记保留在音频后。训练、rollout、reference 与评测绑定同版编译器。模板变更须更新输入缓存身份，按冷暖输入、不同音频和任务、LoRA 版本隔离及训推数值检查验收。
