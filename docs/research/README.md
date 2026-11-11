# 语音多模态 Jev 调研

调研日期：2026-11-09（America/Los_Angeles）。范围：模型、SLU / Agentic Speech 数据集、直接音频决策研究和可复用工程方案。

**当前主线已转为清理后的 SLURP 与本地 Qwen3-ASR 端到端训练准备。** actor 直接消费音频与任务定义；FSC 保留为简单控制，已有 Omni 冻结基线及教师提示优化结果保留，停止扩展 Omni 验证。SLURP clean-v7 元数据和 TRAIN 来源划分已经完成，但语义角色定义、规则族划分与可训练 release 尚未验收，不将受影响标签用于训练/奖励/模型选择。ASR→独立文本模型不进入执行路线，不新增模型下载。

2026-11-11 工作基线：[16 项数据集与指标指南](08-dataset-metrics-working-guide.md)；[单问、多问与推理学习策略](09-query-decision-learning-guide.md)。前者统一任务边界、指标与横比条件，后者分析原子决策与联合判决。用户最新要求已固定为[可配置思考与非文本判决](10-configurable-thinking-decision-contract.md)，取代文本 CoT 实验路线。意图/类型作为 Jev 主范围及具体学习策略仍是建议，不冒充用户已确认或工程已实现。

2026-11-10 更新：已审阅并锁定 [16 项资源的 58 条学术基线登记](07-academic-baseline-lock.md)；后续按 [横比协议](../../experiments/academic-baseline-comparison-v1.md) 分开报告原生任务、派生 Jev 与真实交互。历史综述的资源状态以较新的工程凭证为准。

## 阅读入口

最新执行约束见[决策链—数据集协同设计](11-decision-chain-dataset-codesign.md)：本地学习集覆盖、SFC 轮次捷径、可证伪对照及候选链准入。先验证数据能承载方法，再扩大训练链。

| 文档 | 内容 |
| --- | --- |
| [模型与接口](01-models.md) | Qwen ASR 的最新公开版本、内部结构、适配点、替代模型与运行限制 |
| [数据集与指标工作基线](08-dataset-metrics-working-guide.md) | 16 项资源的角色、Jev 边界、主指标、质量限制与执行顺序 |
| [可配置思考与决策契约](10-configurable-thinking-decision-contract.md) | 思考要求、触发/停止策略、非文本判决复核及 RL 对照 |
| [单问、多问与推理策略](09-query-decision-learning-guide.md) | 原子监督、联合决策、依赖状态、CoT 消融与成本评价 |
| [数据集与任务](02-datasets.md) | SLU、语音函数调用、完整交互、声学与语义诊断；语音来源、划分、指标、许可 |
| [学术研究与已有方案](03-related-work.md) | AudioJev、Qwen-Audio-Jev、SFC、Jarvis 等直接相关工作，以及可迁移的方法 |
| [方案比较与研究切入点](04-recommendations.md) | 研究问题、候选架构、必要对照、分阶段验证和否定条件 |
| [来源与核验方法](05-sources.md) | 一手来源索引、检索边界、版本差异和未解决事项 |
| [模型版本清单](model-candidates.json) | 已读取的模型库 revision 和代码 revision；不是安装依赖锁 |
| [数据核验记录](dataset-audit.json) | SLURP 官方清单计数、Audio2Tool 卡片差异和评测仓库版本 |
| [本地资源与清理](06-local-assets.md) | 共享路径、下载校验与清理范围 |
| [来源请求记录](sources.json) | URL、实际访问时间、响应状态、内容哈希、阅读层级 |

## 最重要的发现

1. **“音频直接产生 Jev 决策”已有直接先行工作。** AudioJev 已发布 Qwen2.5-Omni-3B 衍生权重与推理代码；Qwen-Audio-Jev 已研究专门的 Choice / Noul / Score 头；Prosodia 和独立冻结骨干原型也已存在。不能将加一个分类头、跳过文本生成或候选排序一致性单独声称为首创。
2. **目前从 Qwen 官方公开权重清单核验到的最新专用开源 ASR 主线是 Qwen3-ASR。** 0.6B / 1.7B 与 `-hf` 格式均有具体 revision。更新的 Qwen-Audio-3.1-ASR-Flash-Next 出现在论文中，但本次未核验到其官方可下载权重；同名社区模型也不能替代这一核验。

Hugging Face 授权入口和安全登录步骤见 [hf-auth.md](hf-auth.md)。
3. **Qwen3-ASR 有适合改造的内部接口，但不是通用音频指令模型。** 技术报告 §2.2 明确将其后训练为不跟随一般自然语言指令的 ASR-only 模型。必须检验任务定义能否重新进入决策，而不能只修改 prompt 后假定已经具备 SLU。
4. **SLU、语音函数调用与 Agentic Speech 是不同证据层级。** 意图准确率、工具名/参数匹配、工具执行后的任务成功、实时纠正与取消，必须分别评估。静态数据上的 JSON 合法率不能证明智能体可靠性。
5. **当前算法问题是业务关系奖励能否带来真实任务收益及迁移。** 在相同初始策略、监督来源和预算下比较继续 SFT、独立决策奖励和关系奖励；联合验证定义变化、声学证据、开放参数、缺失状态与完整任务正确性。文献已经涉及其中多项，是否构成创新仍取决于同条件对照及完整复现。详见 [Omni RL 设计](../../proposal/03-omni-sfc-rl-design-v1.md)。

## 已修正的参考资料假设

- NGA 对 Qwen3.5 / 3.8 混合模型的分支隔离分析不能直接套到 Qwen3-ASR：当前核验的 ASR 解码器为 Qwen3 attention 模型。
- Omni 历史调研中“未发现真实人声的工具任务”不能作为当前结论：FDB-v3 已发布真实人声、自然不流畅表达和多步工具调用数据。
- Audio2Tool 论文的“约 30,000 queries”与当前模型库卡片的 **16,843 queries / 36,421 audio files** 不一致；运行时须以固定 revision 的实际清单为准。
- “顺序校准”是对候选排列的概率稳定性约束，不等于概率与真实正确率相符；“一次前向”也不等于任意多问题免费。

初始调研包含公开资料阅读、源码静态检查和本地模型/语料准备，本身没有运行训练、模型推理或付费服务。后续工程已产生独立推理与验证记录，具体结果以 [学术基线锁定](07-academic-baseline-lock.md) 所引凭证及对应实验文档为准；作者报告、工程诊断和本地完整复测分别标注，不能互相替代。
