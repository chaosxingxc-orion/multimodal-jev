# 两阶段简单任务工程方案

**当前生效约束：正式实验总预算四卡（Stage1 DP4，Stage2 DP2 + 两个TP1推理实例），仅简单任务。最新配置与预算见[四卡基线](context-four-device-simple-baseline-v1.md)；下方八卡/6+2说明为历史方案。
最新配置扩展：`configs/experiments/context-two-stage-v0.0.8.json` 为两阶段分别声明 LoRA，并通过 `stage2.reuse_stage1_adapter` 显式选择复用或新参数。规则见[分阶段 LoRA](stage-lora-policy-v1.md)。下文原 loss、全量预算和评测点不变，冻结旧包不改写。

2026-11-17 用户决策：Stage1 学习区分性任务，以 InfoNCE 为唯一 loss；Stage2 强化能力，用 GRPO 的排序和概率目标。只做端到端单音频原生意图判决，完整 clean SLURP TRAIN；跨数据集冻结评测为 FSC/MINDS。

用户进一步澄清：Stage1 不称为「SFT + InfoNCE」。输入为音频和候选定义，训练候选区分能力；没有额外的文本生成、token CE 或 BCE loss。现有运行协议中的 `stage=instruction`、`instruction_objective` 是复用训练传输接口的历史字段名，不代表另有 SFT 目标。正在派发的冻结包保留其字节身份，命名解释独立记录在 `configs/experiments/context-two-stage-semantics-v1.json`。

设计版本 `v0.0.4`；工程候选 `v0.0.6` 已通过八卡普通两更新、新进程恢复和真实尾批的独立有限审核；未发布 rc、未启动正式全量训练。有限审核不代表完整训推一致性或效果收益。旧 pairwise+NLL Stage1 的成绩不是本方案成绩；旧 Stage1 reference 不能替代新 InfoNCE Stage1 的输出。

## 模型和训练协议

同一 Qwen3-ASR-0.6B、q/v LoRA rank4 alpha8、共享一层 Transformer head。执行 BF16；FP32 master、Adam、loss；只保存 LoRA/head、必要 optimizer/marker，base 用 hash 引用。

| 配置 | Stage1 | Stage2 |
| --- | --- | --- |
| 方法 | 区分性任务学习，唯一完整候选 InfoNCE loss | 双信用 GRPO，排序 + 存在性 |
| 训练 DP | 8 | 6 |
| 推理服务 | 0 | 2 个 TP1 实例 |
| 每 DP 副本有效 batch | 64 | 64 |
| global batch | 512 | 384 |
| micro / 累积 | 8 / 8 | 8 / 8 |
| 每轮更新 | 80 | 106 |
| 初始预算 | 2 轮，160 更新 | 2 轮，212 更新 |
| 每轮尾部 | 65 真实 + 447 零权重 | 193 真实 + 191 零权重 |
| 学习率 | 前 8 更新线性 warmup 到 1e-4，此后恒定 | 重置 Adam，1e-4 |
| reference | 无 | 本轮 Stage1 step160，固定 |

两轮预算、micro8、温度1、warmup5%、学习率1e-4是明确记录的初始工程选择，不声称用户指定了这些最优值。batch 由 `64 × devices/(TP×PP)` 推导。设备地址和卡映射在本地宏绑定，不能直接把表格当作已获资源的运行状态。

InfoNCE 使用所有 60 个候选，正确意图为正例，其余为本任务的互斥事件负例：

`L = mean_real_queries(logsumexp(z / temperature) - z_positive / temperature)`。

softmax 仅用于训练损失，部署仍输出各候选 sigmoid(prob)。完整候选单正例下，此 InfoNCE 与候选 softmax 交叉熵数学等价，应如实命名为监督式候选对比学习。没有 batch 内标签采样，没有转写文本替代音频，不把原生单标签负例解释成真实世界所有意图都不存在。

InfoNCE 对所有 logit 加同一常数不敏感，因此不能确定绝对 sigmoid 概率。按用户选择第一版不额外加 BCE；同时记录 NLL/Brier、空集/多选率、set-F1@0.5 和 top1。不能声称 InfoNCE 单独完成概率校准，也不能因概率质量差而静默改输出为 softmax。

Stage2 沿用 `dual_credit_grpo_floor_clipped_v1`：G16、clip0.2、std floor0.25，排序/存在性 RL 各0.5；另加0.1 pairwise softplus、0.3 binary NLL、0.01 Bernoulli KL。不是 RL-vs-SFT 竞赛；辅助监督贡献不全归因于 RL。

## reference 结果缓存

Stage1 固定终点同时作为 Stage2 actor 初始化和 reference。先批量计算全部 40,513 条训练输入的完整候选概率，审核后发布不可变缓存。训练加载一次，之后直接读结果；不保留在线 reference 补算路径，不将变化中的 behavior/动作/优势当作可复用 reference。

绑定策略/base、实际输入张量和 marker、候选顺序、processor、数值 runtime、数据发布与完整样本映射。缓存缺失或身份不一致在 rollout 前失败。Stage2 checkpoint 保存缓存 manifest 与身份，恢复不能换成另一缓存。固定 reference 贯穿两轮，不再实现上一版讨论的周期刷新或 EMA。

`context_reference_dataset.py` 提供完整性验证和一次加载；`context_cached_reference_session.py` 适配现有 session；`context_cached_reference_trainer.py` 在共享 DP6 trainer 上绑定完整人口与 checkpoint 契约。当前这是可测试接口，完整 NPU 建库/封包/准入串接仍待完成。

macro384 一次交给推理池，按32条 rollout 分给一个实例；服务上限32并受 token/audio 内存约束。全 macro 完成后才交训练消费。建库、校验、加载和模型切换成本单独报告，不能把离线成本隐去。

## 实现与验证

- 配置：`configs/experiments/context-two-stage-v0.0.4.json`，无个人机器路径。
- Stage1：`context_infonce.py`、`context_warmup.py`，在 pinned verl actor 与现有训练循环上接入 InfoNCE、DP8 和 warmup，历史运行包不变。
- warmup 由绝对更新 cursor 推导；恢复先匹配最后已提交更新的 LR，再设置下一步 LR，避免重启 warmup。
- 固定镜像 CPU 30 项通过，容器 `bd3bb78b…` exit0、独立收集通过。覆盖 InfoNCE 梯度、DP8/micro归约、全 padding rank、warmup Adam 恢复、DP8/DP6预算和生成训练器绑定。报告 `/srv/multimodal-jev-xingchao/operations/context-two-stage-cpu-v0.0.4/report.json`。
- reference 缓存测试验证错误输入/版本/顺序/内容/人口拒绝、真实 GRPO session 只发 behavior 请求，以及缺失在推理前失败。CPU 证据不等于真实模型建库完成。
- 本次本机 CPU 合并检查共41项通过（上述30项 + reference相关11项）。其中新增reference集成尚未执行固定镜像/NPU验证。正式训练生成器存在“有限窗口先替换checkpoint段，正式窗口找不到锚点”的组合问题；新缓存候选在未替换保存段的共享integration上组合正式窗口，保留预登记稀疏checkpoint，未改历史冻结包。

本地 `.venv` 没有 torch；未改变其环境。先用现有系统 CPU torch 排错，再通过固定训练镜像 CPU 验证；后者才是当前镜像兼容证据。

## 剩余准入和已核对阻塞

1. 绑定经过审核的初始 joint snapshot、数据、DP8 全量 schedule；封装新候选及本地宏。
2. DP8 真实 NPU 两更新、尾部归约、导出和新进程恢复；确认学习率及 LoRA/head 保存。
3. 完成 Stage1 全量训练与预登记曲线，再用其固定终点建库。
4. DP6 + 双推理服务的只读 reference 路径做有限验收和恢复，再发布 rc，启动全量 Stage2。
5. 源 selection 完整曲线评选后冻结策略，运行 clean TEST/FSC/MINDS；目标数据不参与更新、阈值或选模。

原 MINDS batch32 诊断的已收集报告：两种声学路径在声学层全部 exact，单条最终输出 exact，整批32条中31条最终输出不同，最大 logit0.34375、prob0.0440187454。说明当前诊断定位到声学之后的批处理路径；尚不能断言是 LoRA 加载问题或已修复。完整逐层独立复算尚待收口。

历史 DP6 v9 容器现场 inspect：`1c7a3a13…` 已 exited，ExitCode137、OOMKilled=false、FinishedAt `2026-11-17T08:31:10Z`；不能凭137断言 OOM，也不能称其为成功训练。旧证据与运行包继续保护，不将它冒充新两阶段实验。

普通监控900秒；效率对照最多100步；所有新包独立版本，经过审核后再淘汰旧开发候选。

## 2026-11-17 工程验收补充

固定镜像 CPU 新增原生 verl DP8 测试，32 项全部通过、无跳过，并已独立收集。实际执行 full512 / tail65、micro8 梯度累积及新进程 Adam/warmup 恢复；这里只证明合成 CPU 路径，不等于 NPU 数值对齐。报告：`/srv/multimodal-jev-xingchao/operations/context-instruction-native-cpu-v0.0.5/report.json`。

初次 CPU 测试因测试配置遗漏 `jev_context_parity` 失败，已补齐；原失败报告保留在相邻 `v0.0.4` 目录。原生输入缓存改为只读、缺失拒绝，候选 `v0.0.5` 同时修正旧写盘检查与只读挂载的冲突。DP8 全量 schedule manifest 为 `3122deea2684198002dd57ec95c760bff999a3f364c72c84d5d1d30846e1d495`。

NPU 派发首先在主机预检时遇到初始 policy 的700权限，尚未创建容器；固定脚本只修复明确引用的6个策略资产读取权限，哈希不变，收据在 `/srv/multimodal-jev-xingchao/operations/context-instruction-npu-v0.0.5/policy-access-repair.json`。后续有限验收使用同一未执行候选，独立派发记录位于该目录的 `launch-after-access-repair/`。它仅运行两更新；终态后必须 collect + 独立审核，尚不能声明正式训练或效果收益。


## 2026-11-17 终态审核与 rendezvous 修复

`v0.0.5` 容器 `4144d1c0…` 已于 `2026-11-17T14:40:38Z` 退出，exit1、OOMKilled=false。独立收集6个文件，源/pins无变化，没有 rank 结果或模型更新。根因：`torchrun --standalone` 向 TCPStore 发布容器 hostname，`network=none` 下无法解析，两个300秒连接超时。证据：上述 `launch-after-access-repair/` 的 `*-collected/supervision/actor.log`、inspect、transport。新开发包 `v0.0.6` 使用单机 static rendezvous、`127.0.0.1:29500`、零自动重启，保留隔离网络；模型、loss、调度、warmup均不变。

新增 `audit_context_infonce_training.py` 独立核验完整源调度、实际标签/slot、FP32 master与Adam、warmup LR、零在线缓存补算、BF16导出与固定marker。它显式不宣称 NPU 梯度等价、实际长度重算、fresh-process恢复或效果收益。`context_infonce_recovery.py` 只接受审核通过的两步父运行，复用完全相同源码，在新容器恢复 step1 并重新执行 step2，再比较8个rank全部 master/Adam与BF16 policy。原始父运行未通过，故尚未派发恢复。

本机新增审核/恢复/启动测试合计22项通过，包括拒绝错误LR、目标、rank、cursor、backbone checkpoint、损坏Adam和重复slot，以及rank7 Adam差异检测。该结果是CPU工程证据。固定镜像静态rendezvous测试的精确容器句柄为 `66753b89…`；验收报告与是否派发NPU以 `/srv/multimodal-jev-xingchao/operations/context-infonce-rendezvous-cpu-v0.0.6/` 和 `context-instruction-npu-v0.0.6/` 的实际产物为准。

静态loopback固定镜像测试已exit0、独立收集通过（1项，无跳过）。`v0.0.6` 已派发两步NPU有限验收，精确容器 `96308307a5861e0bc89d98083265a2606a2049376877e6cf7a7067ce3a9f1102`；启动观察为 `running`，该观察不代表更新完成。新候选后续仍须真实产物审核与fresh-process恢复。


## 2026-11-17 DP8 真实更新及恢复验收

`v0.0.6` 两更新已exit0并独立审核通过，报告 `/srv/multimodal-jev-xingchao/operations/context-instruction-npu-v0.0.6/infonce-audit.json`。每步512真实样本，step1改变56个LoRA张量/12个head张量，step2改变112/12；128个可训练张量共6,878,209参数，8份FP32 master精确重组BF16导出，固定marker保持。最慢rank步耗时47.5183s、36.7008s，仅短验收，不作稳态吞吐或效果结论。

新容器 `02920cd4…` 从step1恢复并重放step2，exit0且独立审核通过。8份master/Adam、实际batch、BF16 execution和policy全部与连续训练exact。报告 `/srv/multimodal-jev-xingchao/operations/context-infonce-replay-v0.0.6/replay-audit.json`。该证据解决fresh-process恢复，尚不证明NPU梯度等价或源任务收益。

随后准备真实尾批验收：完整调度索引79/80，真实样本65/512，第一批447零权重padding，优化器为诊断cursor0/1，绝不冒充正式step80/81。候选与产物位于 `/srv/multimodal-jev-xingchao/operations/context-infonce-tail-v0.0.6/`；本机6项尾批选择/拒绝测试已通过，固定镜像及NPU准入以相应独立报告为准。正式全量训练尚未派发。

reference建库增加 `context_reference_builder.py` 与配置入口 `build_context_reference_dataset.py`：按实际TRAIN清单编译并绑定每条输入，以macro批量调用已启动的冻结服务，再发布sample→request映射和概率表。重复输入可共享结果但不得丢样本；错序、缺失、运行时/策略变化拒绝。严格要求本轮完整InfoNCE终点、同base/数据版本；不接受短验收checkpoint。接口已有CPU测试，尚未进行实际Stage1终点建库或Stage2训练。输入预处理缓存与reference结果缓存仍为两类独立资产。


尾批固定镜像6项通过，独立报告 `/srv/multimodal-jev-xingchao/operations/context-infonce-tail-cpu-v0.0.6/report.json`。随后派发八卡两更新真实音频尾批验收，容器 `507cddc33cdc9cba9c2219a163b245cf84d3c6eaef846feededd367a42d9760e`，启动时现场确认running；未完成独立审核前不发布rc、不启动正式训练。

正式Stage1封包准备入口 `context_infonce_formal.py` 已实现：只接受同runtime的普通prefix、fresh replay、真实tail审核，保留InfoNCE唯一loss及warmup，派生160步和0/16/…/160评测点。当前仅封包逻辑，正式部署/全曲线评测串接仍待完成，不能把配置生成称为训练启动。

尾批随后已完成独立收集和审核，报告 `/srv/multimodal-jev-xingchao/operations/context-infonce-tail-v0.0.6/tail-audit.json`，状态 `finite_infonce_tail_audited`。两次诊断更新分别为65真实+447零权重padding、512真实，八份master/Adam及BF16导出审核通过。这不是正式训练第80/81步，也不单独证明NPU梯度等价或实际长度分组重算。普通prefix、fresh replay和真实tail三项有限审核已齐备；正式部署与预登记全曲线串接仍待完成。

## 正式部署和分阶段 LoRA 工程补充

正式部署入口已增加 validate/launch/detach/audit，复用普通prefix、fresh replay、tail的原始代码及准入证据。审核器支持每16步保存：每步检查真实人口、LR、cache、optimizer记录和耗时；只在checkpoint处独立重组master、恢复Adam并验证BF16，明确区分运行日志与独立张量证据。50项固定镜像CPU检查通过，报告 `/srv/multimodal-jev-xingchao/operations/context-infonce-formal-cpu-v0.0.7/report.json`；原普通/尾批真实产物复审通过。

新增LoRA功能随后独立验证。首轮固定镜像CPU共65项、3项失败：一个动态生成器依赖未封入测试包，两个测试模型包装器未声明SDPA支持；实际训练未派发，失败证据保留在 `/srv/multimodal-jev-xingchao/operations/context-stage-lora-cpu-v0.0.8/report.json`。已修正测试封包与包装器，新的显式复用开关测试使用独立v0.0.9包，结果以其独立报告为准。正式源训练和11点评测串接仍未完成，不把封包称为启动。

最新四卡预算已覆盖旧八卡正式封包；旧rc2/rc3/rc4仅准备、未派发，不再作为可启动正式配置。新增正式入口四卡预算保护。当前主线看四卡基线文档及v0.0.11配置。
