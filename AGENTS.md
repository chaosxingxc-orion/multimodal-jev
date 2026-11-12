# Repository Guidelines

## 范围与交流

本项目研究语音多模态 Jev 的设计与实现，主要参考 Qwen 系列最新开源 ASR 模型，
以现存 SLU 和 Agentic Speech 任务及数据集为研究基础。默认中文交流与报告；代码标识符使用英文。
当前为基线、数据协议与 RL 接口实现阶段，模型/数据版本见 docs/research 和 docs/engineering；区分已验证 CPU 功能、待验证 NPU 路径和正式训练结果，不把规划描述成已完成能力。
当前执行优先级：冻结基座推理基线 → 参考 omni 的远程教师 TFRL 提示词优化 → xx-jev 消费数据及同类优化。模型权重保持冻结；后续权重训练队列暂不派发，既有运行证据保留。教师仅消费学习集反馈，选择集用于候选筛选，测试集不参与优化。
2026-11-11 用户指出 omni 已发布 SLURP 标注缺陷：受影响标签及指标不得作为训练、奖励、教师提示优化反馈或模型选择依据。在 clean-v7 清单与本项目数据逐条对齐、重新发布并验收前，SLURP 全部指标仅用于诊断，不进入优化。既有冻结推理及原始分母保留，不原位修改 gold；清洗后指标独立标注，不能把删样本带来的变化称为算法收益。详细来源和限制见 docs/engineering/slurp-quality-hold-v1.md。
用户已明确要求端到端任务：推理必须直接消费音频与任务定义，由同一个语音多模态模型输出任务结果。禁止派发“语音转写→独立文本语言模型”或参考转写替代音频的主任务实验。既有串联路线只保留历史诊断，不计入端到端基线、TFRL提升或跨任务泛化。远程教师可离线分析学习反馈；在线actor仍直接听音频。Qwen-ASR任务指令能力不足时保留失败证据，研究直接音频任务模型/适配/typed读出，不以串联文本模型绕过。
优先遵循用户当前指令；参考工程的机器配置、角色分工和历史运行状态不自动适用于本项目。

## 开发

2026-11-12 任务先行设计稿：docs/research/24-task-definition-freeze-draft.md与configs/tasks/jev-task-design-v1.json登记I_class/I_present/S_present/B_bound真值、Final关系及迁移条件。55槽位卡仍全部draft；S需区分explicit_filler和允许未赋值角色激活，现有filler模板未覆盖新契约，不冒称已实现。参考NGA定义消费/稳定配置及omni槽位角色边界，来源hash已归档。尚无完整定义/监督准入，不训练。上轮未提交的恢复入口/驻留概率检查修改暂存为工作树草稿，不继续派发或冻结成正式工程版本。

2026-11-12 最新用户要求任务→工程→训练顺序：先完成Q/任务/监督/Final设计，再锁定统一工程基线与实验配置，最后启动训练。撤销processor完成后自动恢复Stage1的安排；不派发新研究训练，已有只读检查正常归档。登记PPO-critic、GRPO、RLOO三种通用配置，同一typed-action工程底座；现有实际RL是无critic的精确有限候选边缘PPO定制变体，最近两步NPU只是CE。step-wise先研究语义节点信用，不能把并行Span当时序状态套GAE。configs/experiments/rl-baseline-design-v1.json明确training_dispatch_allowed=false，尚未工程锁定。详见docs/research/23-task-engineering-rl-baseline-lock.md。

2026-11-12 prefix导出与一致性更新：发现并修复LoRA键名丢点；保留原快照，从四rank FP32分片独立重建112张量，实际PEFT重载精确。修复adapter身份85120a606c313c142e11c7444a1242cb526898bf9fa82251423b891771e7b1ac；prefix-parity-v3实际exit0，256输入HF/vLLM最大logprob误差9.54e-7、argmax变化0。无驻留FSDP输出证据，不夸大。全量恢复入口已实现但未NPU验收/派发，full processor仍运行。可迁移Q增加固定菜单与显式正/负/缺标映射；84项相关测试通过，未接入新训练。详见docs/engineering/slurp-instruction-recovery-and-predicates-v1.md。

2026-11-12 最新用户明确核心是可迁移判决Q：先分析同任务跨数据集共同的逻辑判决要素，再定义“query是否表达定义中的意图”“是否包含定义中的槽位类型”等参数化Q，训练并检查Final收益。不能只提供笼统思考要求。规约供给概念/关系定义，Q算子固定、实例定义由任务配置提供，禁止gold选题。单标签成员与多意图语义存在性区别登记，缺标不能转false；先I/S，绑定/条件/状态依数据监督扩展。首版decision_questions仅编译输入未准入新训练。详见docs/research/22-transferable-decision-questions.md。

2026-11-12 最新用户定位：设计语义规约及思考决策范式供模型理解、参考和使用，以服务Final；不是让模型自己生成思维链。任务级规约定义概念边界、证据、绑定/组合、优先级和typed读出；具体证据由模型从音频获取，不按样本主题预选问句。保留一次交付可配置Span，未必每条规约都对应监督头；过程缺标不造reward。两阶段主线依然有效，现有全量视图仅是工程起点，不冒称完整规约学习。详见docs/research/21-semantic-decision-specification.md，优先于历史CoT定位。

2026-11-12 最新用户否定query-specific问句主设计：撤回按闹钟/日历、设备等主题模板包推进训练的建议，候选配置新增do_not_dispatch处置。固定模板不等于通用问题；禁止先猜当前样本主题再选择前置问题，禁止用gold/参考文本/离线复杂度选题。问题构造只依赖部署时已知任务定义、完整候选、本体属性和用户思考配置。通用过程问题缺标不得虚造奖励；显式属性分解与隐式计算候选均须验证，不能预先声称推理收益。现有全scenario/action视图是全本体辅助分类工程控制，不是已证明通用CoT。详见docs/research/20-query-independent-thinking-contract.md。

2026-11-12 正式学习人口问句审计：40,513录音中26,834单组件可消歧、13,679需scenario/action联合；闹钟/日历两问支持6类7,422条联合消歧，设备两问支持8类2,808条。不同问句恒答outside_family可达81.68%–96.37%，过程必须报告族内指标。新问句仍candidate，不原位改冻结Stage1日程。详见docs/research/19-discriminative-question-working-guide.md；10项测试及独立人口/分区复算通过。四卡prefix实际exit0、四rank两步完成，记录见docs/engineering/slurp-instruction-prefix-v1-observed.json，尚未独立checkpoint/重载/跨后端验收，不等于Stage1或效果完成。

2026-11-12 原生混合actor进展：`span_hybrid_actor`经verl注册loss接口保持原生update_policy负责累积/backward/Adam，真实四rank CPU连续更新与恢复已通过，最大梯度误差2.98e-8。全量两轮processor预检`slurp-instruction-processor-full-v2`及真实四卡global128两更新前缀`slurp-instruction-dp4-prefix-v1`已派发；这是新LoRA的执行验收，不是完成Stage1。核验实际launch/container及独立验收后才能继续全量。prefix入口尚无恢复继续功能，不能直接宣称支持完整常驻训练。详见docs/engineering/slurp-native-hybrid-actor-v1.md。普通状态检查仍至少600秒。

2026-11-12 两阶段主线数据/目标进展：`datasets/slurp-jev-benchmark/official-clean-intent-v1`已发布，manifest `38e4273d9328e4f02db081315e46f97a17628b974be7ba175254882e95421482`。官方clean标签依据，learning 40,513录音、selection 9,226录音，分别9,181/2,146来源组、均60类；已知pending整组隔离共647录音。49,739条音频已重新hash/解码，通过独立标签/人口复算，无语义复听声明。使用新的slurp_official_training接口，不关闭旧reviewed准入校验；selection禁止优化。混合CE+verl有限PPO已通过固定镜像四rank CPU梯度/Adam恢复，尚未接入native actor完整生命周期或NPU音频混合训练。详见docs/engineering/slurp-two-stage-data-and-loss-v1.md。

2026-11-12 用户补充要求全量解构数据集的简单/复杂样本，设计判别式问句改善Final。复杂度分为本体消歧、语言证据、声学、模型经验四维，不用长度或组件数冒称真实推理难度；按gold挑问句禁止。全量learning分析与固定问句候选见docs/research/18-slurp-complexity-and-discriminative-questions-v1.md。规则投影只监督任务属性，否定/指代/可听证据需独立标签，缺标mask。

2026-11-12 最新用户调整（优先于下述小样本与纯RL安排）：主训练回到规模化SLURP TRAIN，采用两阶段：先学习音频指令/问题/候选绑定的消费能力，再学习任务能力；第二阶段使用RL loss加可逐渐衰减的SFT分类loss。不新增RL/SFT胜负对照。“不比较RL与SFT”不禁止辅助监督损失。89来源先导只作历史工程诊断，不再作为主学习人口。旧小样本控制器已停止继续派发，已启动容器不强杀，不把未完成P曲线记为完成。使用新的版本化发布协议消费clean-v7官方意图标签，并隔离已知缺陷、冲突及项目已标pending来源；明确official-clean与逐条reviewed区别，不伪造审核状态或原位修改旧optimization_allowed。类型缺标不能补false。保持端到端音频、一次Span读出、BF16、175卡4–7与来源隔离。执行方案见docs/research/17-slurp-two-stage-training-v1.md；新混合loss、规模化loader与常驻循环尚需实现和真实验收，不能声称已经启动大规模训练。

2026-11-12 原生意图新增受限准入：预先固定120条学习来源经参考文本逐条语义审核，89条意图标签准入、31条pending，覆盖56类，候选仍为60类。未复听音频，不代表全本体或基准评测认证；不准入槽位类型。发布与限制见 docs/engineering/slurp-intent-reference-admission-v1.md。D/Q/P过程标签仅由已审核joint intent的canonical scenario/action投影，不称独立过程标注或已验证多跳推理。新产物仍需processor/backend验收后才能派发。

2026-11-12 用户明确三条研究主线：简单任务 RL Loss/Reward；预先拆解的复杂任务中多步显式结构判决、最终综合判决的 Loss/Reward 及相对直接问题的增益；相似任务跨数据集配置迁移，FSC 为首个指定例。显式结构保持一次填充 Span 契约，不自动转成文本 CoT/外部解码。允许直接问题 RL 与结构问题 RL 对照，不新增 RL/SFT 效果对照。FSC 当前用于冻结源域模型的配置迁移，不恢复历史 FSC SFT 主训练。方案见 docs/research/14-rl-decision-credit-and-transfer.md。

2026-11-12 用户进一步调整验收目标：不比较 RL 与 SFT 的表现，不再以 RL 超过同预算 SFT 作为条件。主线验证固定 RL 起点后，任务指标随 RL 更新的持续提升；固定评测名单、本体、奖励与预算，报告预定检查点的完整曲线和退化情况。已有运行正常收尾、历史结果保留；不新增 SFT 对照训练。SFT 来源的初始 adapter 如被采用，仅如实记录为初始化。Omni 仍为任务能力定位参照，跨任务泛化目标保留。详见 experiments/slurp-asr-jev-dp4-lock-v1.md。

最新主线锁定（2026-11-12 UTC）：用户要求 Qwen-ASR 在 Jev 类评测上对标基座，参考 omni 的 SLURP 基线。使用本地 Qwen3-ASR-0.6B、BF16 LoRA、175 物理卡4–7的四卡 FSDP 数据并行；rollout 使用已通过一致性检查的 vLLM TP1。该决定优先于历史冻结权重阶段，不切换到30B训练。主评测是意图分类和不提供意图的槽位类型集合；时间三布尔/规则任务仅为独立扩展诊断。既有训练及自动评估正常收尾，取消的单卡成对奖励队列不得重启，四卡可用后优先恢复与连续训练验收。已有四卡一步不等于完整 verl trainer；全量本体也未全部准入训练。协议见 experiments/slurp-asr-jev-dp4-lock-v1.md，参考源见 docs/research/omni-slurp-jev-reference-lock-v1.json。旧 clean-v7 总体暂停按已完成的逐标签准入收窄：仅审核通过的监督可训练，不能把数据清洗对齐当作所有缺标已解决。

最新精度约束：用户要求 BF16 训练并保证训推一致。Span 主线的基座、LoRA 执行/导出和输出投影使用 BF16；显式关闭 PEFT adapter 自动升精度，vLLM lora_dtype=bfloat16、head_dtype=model。FP32 候选归一化/损失及优化器 master/state 保留稳定性。两侧均使用同一未合并适配器，禁止把 FP32 训练 adapter 仅在 rollout 降为 BF16 后声称一致。通过保存重载及跨后端全候选/联合概率 gate 前不准入 RL；同 dtype 不等于已通过一致性验收。

用户新增输入与调试要求：允许可配置 preemphasis、首尾静音处理、音量处理和语速增强。保留 raw 对照，记录处理后音频身份，actor/rollout 使用同一波形或特征；速度与保音高 tempo 不混称。调试训练要求分类均衡：检查单标签及联合标签分布，并保证当前三任务每个 batch 正负均衡；不修改原始发布、不补造缺失监督、不改变测试分布。见 docs/engineering/audio-preprocessing-v1.md 和 experiments/slurp-native-spans-balanced-v1.md。

最新用户纠正（优先于下文历史策略）：主任务是一次填充的 Jev 决策 CoT：输入音频及 `Q1,<SPAN1>,Q2,<SPAN2>,Final Q,<SPAN3>` 完整问题模板，由同一模型一次性输出各 Span 的 typed value/state。研究结构化 CoT 或模型内部隐式思维链，不采用 ReAct、外部循环触发/复核/停止控制器，不以自回归文本解码作为新任务的实现目标。所有前置问题在输入时给定，最终答案由模型读出。既有单问 JSON 生成实验仅为历史工程证据，不能冒称已实现 Span 一次填充。当前契约见 docs/research/12-single-pass-span-cot-contract.md；其内容取代 09/10/11 中把有界复核和学习停止当主线的安排。

2026-11-11 用户明确：所有Jev模型应接受可配置的思考要求，通过推理时的触发和决策提升能力，不将触发文本推理作为研究路线。配置化思考是共同模型契约；单问/多问是独立任务组织维度。优先研究要求条件化决策、结构状态和有界复核，具体机制及收益待验证，禁止把文本CoT消融当作当前主线。见 docs/research/10-configurable-thinking-decision-contract.md。

2026-11-11 最新数据优先级：用户认为FSC过于简单，明确先用清理后的SLURP开展Qwen3-ASR Jev训练。停止推进FSC先导派发与FSC-v2主训练建设，已有准备仅保留。先完成clean-v7监督冲突隔离、可训练发布、规则族/来源组划分和训练适配；不要将metadata重建等同训练数据已就绪。不得因这一优先级改变而重新引入已确认错误标注或使用TEST优化。

2026-11-11 最新用户转向：当前两项输出就绪后尽快收尾现有验证，开始基于Qwen3-ASR的Jev训练。此指令取代上文长期不派发权重训练的阶段限制；不等Omni完整FSDP/概率验证才开始ASR训练准备。两项输出具体所指已向用户核实，待回复。停止扩展Omni验证范围，交接时逐一核验并停止不再需要的排队监督进程，保留运行证据。默认准备本地0.6B端到端音频+schema生成式LoRA SFT，再接verl/vLLM RL；专用typed head与1.7B作为待明确范围，不声称已有实现。SLURP质量限制继续有效，首轮可用FSC TRAIN，禁止测试反馈优化。

- 可复用 Python 实现放在 `src/multimodal_jev/`，辅助工具放在 `tools/`。
- 配置、研究计划、实验协议和结果分别维护在 `configs/`、`proposal/`、`experiments/`、`wiki/`。
- 使用 Python 3.11+ 和项目内 `.venv`；按变更运行适当验证，业务实现添加有意义的测试。
- 引入参考工程代码前核实来源、版本、许可和依赖，记录复用范围。
- “最新 Qwen ASR”须通过官方来源核验并记录查询日期、模型 ID 和 revision，不凭名称推定版本或用 Omni 替代 ASR 参考。
- SLU 优先尊重原始意图/槽位定义、官方划分和评分；Agentic Speech 保留交互、工具与任务完成维度。
- 不提交凭据、模型、数据集、原始业务数据、checkpoint、完整运行日志或虚拟环境。
- 提交前检查差异；推送前 fetch 并检查远程变化，保留已有工作。

## 实验可信度

- 运行前固定假设、划分、指标、阈值和退化基线，修改协议时明确记录修订。
- 每个不变性目标配套区分性测试；不能仅用聚合准确率证明决策一致性。
- 明确输入模态、证据来源和未知/缺失/冲突状态；测试数据不用于调参。
- 记录代码版本、配置、数据/模型版本、启动命令和结果证据路径。
- 长任务脱离交互会话运行，保留日志、心跳、断点及带退出码的终态文件。
- 交互式实验状态检查间隔至少 10 分钟；自动目标续跑不缩短该间隔，不逐分钟重复汇报。用户主动询问或收到完成/失败通知时可立即核验。该约定不改变既有冻结运行的心跳、监督器与截止时间。
- 报告真实完成量、运行成本和限制；环境准备不代表训练或评测成功。
