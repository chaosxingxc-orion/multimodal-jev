# 实验协议与基础配置

新实验、协议修订及运行配置的变更在这里登记，并从 Wiki 台账链接到对应版本。派发前固定研究假设、对照、模型与数据身份、预算、设备布局、检查点和验收口径。已冻结协议的修订另发版本，保留旧版与变更原因。

- 实验协议及配置索引放在 `experiments/`。执行器实际加载的配置通过相对链接引用原有模板与冻结发布件。机器私有地址和路径由本地宏解析。
- 协议和配置用于登记预期。过程状态、审核收据、分数据集成绩与结论在 Wiki 更新。[实验台账](../wiki/experiment-ledger.md) · [报告目录](../wiki/reports/README.md)
- 在 [pipelines/](pipelines/README.md) 将实验 ID 绑定到已有脚本、审核收据和评测清单。由根目录入口领取台账、执行并回填报告。[五步规约](../wiki/experiment-package.md)

| 入口 | 内容 |
| --- | --- |
| [JEV-209 预登记](stage1-cards-tpo-209.md) | 0.6B r128 知识卡 Stage1 与融合 206、208 的单次 Stage2 |
| [JEV-210 预登记](stage1-cards-tpo-210.md) | 与 209 同一 0.6B、r128 和数据，Stage1 每条录音对比 K 张错卡 |
| [Stage1 双向对比配置](stage1-card-bidirectional.md) | 同语音换卡与同卡换语音；有放回抽样，每个锚点当前 210 每个锚点 14 个配对 |
| [JEV-210 微批短测](stage1-batch-comparison-210-v1.md) | 两档 2 的幂次微批在相同 TRAIN 录音上的吞吐与显存 |
| [JEV-208 预登记](tpo-probability-gap-208.md) | 扩大可信错例覆盖，在 TPO 中加入当前最强错类的概率间隔 |
| [JEV-207 预登记](tpo-actor-pair-margin-207.md) | 验证新构造的源域音频对 + actor 排序项；按新视图推导预算 |
| [简单语音任务基础配置](simple-speech-baselines.md) | 实验范围、数据用途、评测口径、资源限制及两阶段对照设置 |
| [联合 TPO 实验协议](tpo-joint-comparison.md) | 从 Stage1 接续的 JEV-201 和全量单阶段 JEV-204 |
| [学术横向比较协议](academic-baseline-comparison-v1.md) | 公开模型、原生任务、派生 Jev 与交互任务的分层横比 |
| [缓存训练效率协议](context-cached-grpo-efficiency-v1.md) | 工程效率对照的完整计时口径 |
| [历史协议](archive/README.md) | 冻结基线、早期 RL、Span 与延期课题 |
