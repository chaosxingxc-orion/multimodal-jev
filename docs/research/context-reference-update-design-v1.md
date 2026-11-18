# 简单任务 reference 更新与训练集结果缓存

2026-11-17。研究设计，尚未发布为训练基线。用户纠正：references 本身应成为训练集可直接加载的模型结果缓存，不是查找名为 references 的前处理方案。

**后续用户决策已取代本文的动态更新建议**：正式方向为两阶段，Stage1 八卡 SFT + InfoNCE + warmup；Stage2 六卡训练、两卡推理，双信用 GRPO，全程固定 Stage1 模型输出作为 reference 缓存。动态 reference 仅保留为研究背景，不进入当前实现或实验。见[当前两阶段工程方案](../engineering/context-two-stage-v0.0.4.md)。

## 当前实现事实

现有配置仍从完成 Stage1 的策略 `9164eddffd9e3fdb21c060a72827b1a4324f3d45e60eb0d491b589b194db0e66` 开始 Stage2。
`context_dual_credit.hybrid_loss` 的目标为双信用 RL + 0.1 排序辅助 + 0.3 binary NLL + 0.01 Bernoulli KL。
KL 逐候选求和，以 query 为分母。当前 reference 固定，辅助监督与 RL 同次更新。
已归档的 40,513 条 reference 覆盖旧训练输入；原审核没有授予跨 runtime 数值复用资格。

一次启动可以包含多个逻辑阶段；“一次性训练”本身不意味着 reference 必须移动。
但若从不适配任务的 ASR/随机 head 开始联合学习，永久锚定该起点会约束必要的指令学习。
首次 reference 应来自达到预登记指令消费验收要求的监督适配状态，现有 Stage1 可复用。

## 三种不同角色

| 角色 | 用途 | 更新边界 |
| --- | --- | --- |
| 当前 actor | 参与梯度更新，产生新策略 | 每次 optimizer update |
| behavior / old policy | 采样动作以及 importance ratio 分母 | 每批 rollout 固定；不能随着 actor 更新重写旧概率 |
| KL reference | 限制策略偏离某个锚点 | 可全程固定或在外层阶段边界更新，阶段内部冻结 |

GRPO 的组均值/标准差、RLOO 的留一奖励均值是优势估计基线，不是 reference 模型。
PPO、GRPO、RLOO 的估计器差异不强制决定 KL reference 的刷新频率。
更换 reference 会改变优化目标，不能当作仅改变缓存或 I/O 的工程等价优化。

## 已核对的算法与开源依据

1. [DeepSeekMath / GRPO](https://arxiv.org/html/2402.03300v3)，Algorithm 1 与 §4.1.4：外层 iteration 开始时令 reference = 当前 policy，内层 steps 保持它固定；old policy 则在内层采样前刷新。论文外循环还包含 reward model 更新；本项目使用可计算奖励，只借鉴其 reference 更新机制，不声称逐项复现整个算法。
2. [Learn Your Reference Model for Real Good Alignment / TR-DPO](https://arxiv.org/html/2404.09656v4)：研究动态更新参考策略的偏好优化。其直接证据来自偏好/语言生成任务，不是语音 Jev 效果证据。
3. [TRL GRPO 文档](https://huggingface.co/docs/trl/main/en/grpo_trainer)：支持 `sync_ref_model`、`ref_model_sync_steps`、`ref_model_mixup_alpha`。核对的 callback 实际在参数空间插值；不能把文档中的策略混合简写理解成逐样本概率线性混合。
4. 核对日的 TRL `main` 中，GRPO/DPO 的 reference 同步对 PEFT 报不支持；DPO 还明确禁止同步 reference 与一次性 `precompute_ref_log_probs` 同时启用，理由是旧缓存失效。本项目是 verl + 自定义概率 head，需要自己的版本化适配器快照和缓存实现，不能直接照搬开关。

原始网页/源码快照与 SHA256：本地研究归档 `reference-update-20261117/sources.json`；当前绑定目录为 `/srv/multimodal-jev-xingchao/research/`。这是证据位置，不是执行配置的路径常量。未引入上游源码。

## 推荐第一版：监督适配后建立 reference，阶段边界硬更新

统一训练入口依次执行：监督适配或复用已验收 Stage1 → 冻结 reference-r0 → 生成完整 TRAIN 结果缓存 → RL + 监督辅助 → 到预登记阶段边界导出当前 adapter/head 为 reference-r1 → 重新建库 → 下一阶段。

不建议第一版同时加入每步 EMA、独立 SFT 教师和额外多锚点 KL。硬快照保留现有 LoRA rank，易验证、恢复和复用；只保存 adapter/head 与 base 引用。外层更新间隔作为配置，不能直接照抄 TRL 默认 512，也不在本次讨论中视为已经冻结。

如果用户所说“每次 SFT 后更新”指分开的 SFT 与 RL 优化块，可在 SFT 块提交后取快照。如果使用单个混合 loss，一次 optimizer step 的结果同时受 RL 和监督影响，不能称为“纯 SFT 后 reference”。若必须获得纯监督参考，需单独监督阶段或监督教师；后者增加训练成本，不推荐作为首个效率基线的依赖。

移动 reference 约束近期偏移，不保证保留原始 instruct 能力或提升迁移。保留监督辅助并报告对初始 Stage1 的决策/概率变化；FSC/MINDS 只做冻结后迁移评测，不参与 reference 更新选择。源 selection 只用于既定检查点评选，不驱动训练中刷新、阈值或奖励调参。

若需要额外固定锚点，目标可扩展为 RL + 监督辅助 + beta_local KL(current || moving_ref) + beta_anchor KL(current || initial_SFT)，但它是另一个研究配置；初始 SFT 的源域预测也不能独立证明通用指令能力被保留。

## 缓存及恢复契约

- 缓存内容：每条确定性音频/任务输入的完整候选 reference 概率，必要时附 raw logits。输入 processor 缓存另行复用；行为概率、动作、奖励和优势不能跨更新缓存替代。
- 身份绑定：reference adapter/head、base、输入张量/marker、processor/前处理、任务模板及候选顺序、数据发布、数值 runtime；同样本换候选或增强音频必须重建对应项。
- 每个 reference 版本一个不可变 manifest；启动或阶段切换前审核所需 TRAIN 完整覆盖。正式训练只读，缺失即失败，不在 optimizer 热路径补算。
- 发布顺序：提交 reference 快照 → 建库 → 全量完整性审核和所需数值验收 → 原子切换活动版本。所有 DP rank 使用同一版本，进行中的 rollout 不跨版本。
- checkpoint 绑定 reference 版本、manifest hash、阶段/数据 cursor；恢复使用原版本，不静默取最新版本。
- 单任务固定 Bernoulli 候选动作可由完整概率精确计算各动作的 reference logprob 和 KL，因此适合提前建库。这个性质不能直接外推到开放文本生成的任意新续写。
- 更新 reference 后旧结果不再表示新 reference；缓存 EMA 概率不等于模型参数 EMA。对 LoRA 的 A/B 分别做 EMA 也不等于对有效权重 BA 做 EMA。

效率报告必须包含建库、验证和阶段切换成本。每阶段只走一遍数据时，把 reference 前向移到训练前主要消除切换和改善批处理，并不会自动省掉全部前向次数；阶段内多轮或同版本跨实验复用才有额外摊销收益。

## 发布边界

当前仍需解决已记录的 MINDS 训推数值差异，以及新缓存路径的实际更新/恢复验收。本文不放宽准入阈值，不宣称 NPU 已通过、不发布 rc、不自动启动正式训练。确认 reference 语义后，以独立开发版本实现并验收，固定 reference 缓存与移动 reference 算法分别记录，避免把算法变化误称为纯效率收益。
