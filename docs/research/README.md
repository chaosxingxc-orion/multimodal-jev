# 语音多模态 Jev 调研

调研日期：2026-11-09（America/Los_Angeles）。范围：模型、SLU / Agentic Speech 数据集、直接音频决策研究和可复用工程方案。

**当前主线是语义规约驱动的语音决策、RL+衰减CE的任务提升及跨数据集配置迁移。** 使用本地 Qwen3-ASR-0.6B、BF16 LoRA、四卡 FSDP 数据并行与 vLLM TP1；模型消费任务规约和预先提供的决策范式，一次交付typed Span，不生成推理过程。SLURP官方clean标签发布已准入40,513条学习录音，独立内部selection为9,226条；不等于逐条语义复听或槽位监督准入。采用先学习指令/规约消费、再任务RL+衰减CE的两阶段方案，不新增RL/SFT胜负比较。FSC用于冻结权重的配置迁移。工程验收不等于任务收益。

2026-11-11 工作基线：[16 项数据集与指标指南](08-dataset-metrics-working-guide.md)；[单问、多问与推理学习策略](09-query-decision-learning-guide.md)。前者统一任务边界、指标与横比条件，后者分析原子决策与联合判决。用户最新要求已固定为[可配置思考与非文本判决](10-configurable-thinking-decision-contract.md)，取代文本 CoT 实验路线。意图/类型作为 Jev 主范围及具体学习策略仍是建议，不冒充用户已确认或工程已实现。

2026-11-10 更新：已审阅并锁定 [16 项资源的 58 条学术基线登记](07-academic-baseline-lock.md)；后续按 [横比协议](../../experiments/academic-baseline-comparison-v1.md) 分开报告原生任务、派生 Jev 与真实交互。历史综述的资源状态以较新的工程凭证为准。

## 阅读入口

当前入口为[三条研究主线工作基线](15-rl-three-track-working-baseline.md)及[完整 RL 协议](14-rl-decision-credit-and-transfer.md)，输入输出遵循[单次 Span 契约](12-single-pass-span-cot-contract.md)。[组合数据承载审计](16-composition-data-support.md)区分标签分解与多步推理证据。09–11保留设计演进，其外部触发/停止循环不再是当前执行契约。

| 文档 | 内容 |
| --- | --- |
| [同任务跨数据集的判决Q](22-transferable-decision-questions.md) | 最新细化：意图、槽位、绑定等共同逻辑要素，定义驱动监督与Final/迁移收益 |
| [语义规约驱动决策](21-semantic-decision-specification.md) | 最新定位：供给概念、证据规则、决策范式与typed读出，模型理解并使用 |
| [三条研究主线工作基线](15-rl-three-track-working-baseline.md) | 简单任务 Loss/Reward、D/Q/P 过程信用、SLURP→FSC 配置迁移与收益判据 |
| [完整 RL 协议](14-rl-decision-credit-and-transfer.md) | 有限候选边缘 PPO、监督预算、冻结实验与证据边界 |
| [单次 Span 契约](12-single-pass-span-cot-contract.md) | 全部问题预先给定，一次读出 typed Span，不生成文本思维链 |
| [组合数据承载审计](16-composition-data-support.md) | 单组件捷径、组合划分可行性及多步推理结论的限制 |
| [输入未知时的问题契约](20-query-independent-thinking-contract.md) | 当前修订：根据任务定义配置思考，不预知样本主题，不按答案选题 |
| [判别式问句工作指南](19-discriminative-question-working-guide.md) | 人口统计保留；主题模板主方案已根据用户评审撤回 |
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
5. **当前算法问题是 RL 能否持续改善 Final，以及过程信用和配置消费能否带来额外收益。** 固定初始化和来源预算，比较直接问题 D、多问题但仅 Final 奖励 Q、多问题加过程奖励 P；另用冻结 FSC 模板区分权重迁移、模板收益及二者交互。过程标签与计算开销单独披露，不把总 reward 上升或配置可加载当作成功。详见[三条研究主线工作基线](15-rl-three-track-working-baseline.md)。

## 已修正的参考资料假设

- NGA 对 Qwen3.5 / 3.8 混合模型的分支隔离分析不能直接套到 Qwen3-ASR：当前核验的 ASR 解码器为 Qwen3 attention 模型。
- Omni 历史调研中“未发现真实人声的工具任务”不能作为当前结论：FDB-v3 已发布真实人声、自然不流畅表达和多步工具调用数据。
- Audio2Tool 论文的“约 30,000 queries”与当前模型库卡片的 **16,843 queries / 36,421 audio files** 不一致；运行时须以固定 revision 的实际清单为准。
- “顺序校准”是对候选排列的概率稳定性约束，不等于概率与真实正确率相符；“一次前向”也不等于任意多问题免费。

初始调研包含公开资料阅读、源码静态检查和本地模型/语料准备，本身没有运行训练、模型推理或付费服务。后续工程已产生独立推理与验证记录，具体结果以 [学术基线锁定](07-academic-baseline-lock.md) 所引凭证及对应实验文档为准；作者报告、工程诊断和本地完整复测分别标注，不能互相替代。
