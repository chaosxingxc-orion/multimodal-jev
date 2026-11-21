# context-prob-v1：当前任务、工程与算法基线

> 2026-11-18 最新启动入口：[`run_composed_stage2`](../../tools/reports/run_composed_stage2.py)，
> 当前候选v0.0.91，先完成短程可靠性验收，随后执行正式634步训练。
> 用户已取消独立效率基线；效率从正式训练固定窗口采样统计，不额外派发100步。
> 实际入口、冻结证据及状态边界见[启动修复 §7](stage2-startup-root-causes-20261118.md#7-组合式启动与正式训练内效率统计)。
> 本页下方的旧运行状态、旧Stage1 loss和旧GRPO/RLOO对照流程是历史记录，不用于启动本轮实验。

> 文档索引：本页保留 2026-11-15 的发布身份与历史状态。2026-11-17 起的最新范围、算法与
> 效率决策以[工程索引](README.md)和[协作指南](../collaboration/simple-task-rl-engineering-guide-v1.md)
> 为准；当前算法主线为 `dual_credit_grpo_floor_clipped_v1`，旧 GRPO/RLOO 同预算对照是历史发布状态。
> 当前四卡两阶段执行入口为[简单任务四卡基线](context-four-device-simple-baseline-v1.md)：
> Stage1 DP4 InfoNCE，Stage2 DP2 + 两个TP1推理实例；复杂任务及数据集延期，不作为本轮依赖。

本页为执行入口。顺序固定：**任务与比较设计 → 工程及配置冻结 → 有限验收 → 正式训练**。
当前交付重点是全量简单任务的完整曲线、学术参考比较和冻结策略迁移，不以工程检查替代收益。

## 状态与观察入口

普通监控固定每900秒。没有新阶段结果、异常或具体待办时安静等待，不主动触发分析、
重复进程/依赖核验或逐分钟汇报；自动目标续跑不视为新增待办。新证据到来后由固定脚本
生成loss、reward、RL算法诊断及效率报告，沿用现有监控与终态分析，不重复创建任务。
新结果本身不唤起模型；研究性解释和方案调整在明确的研究、开发或异常排查任务中进行。

## 日常任务执行规约

从 Wiki 队列领取实验，在实验目录维护预登记协议与配置。各阶段沿用已发布的运行入口，以同一研究 ID 交付数据构建或复用、训练、评测及台账更新。[五步实验包规约](../../wiki/experiment-package.md)

配置统一使用`${NAME}`引用本地宏，机器路径/主机/设备/本地产物身份集中在不入库的
`.local/config/macros.json`；共享模板及统一解析入口见[本地宏规约](local-config-macros-v1.md)。
冻结运行继续消费原配置，新任务使用解析后另行发布的配置，禁止各工具重复实现宏替换。

日常任务统一使用**一套共享工程基座、固定脚本、版本化配置**。定时器和终态事件直接
调用程序，不调用模型人工分析、挑选文件、生成临时命令或串接流程。模型用于明确的
研究、开发和异常排查；形成可复用规则后固化为脚本及配置。

| 日常任务 | 必须固定的配置及行为 |
| --- | --- |
| 本地/远程清理 | 明确路径允许范围、保留规则及归档位置；先输出候选清单，校验归档后删除。保护共享模型/数据、最终结果、配置和运行/排队依赖；保留最终checkpoint，仍被评测或恢复依赖的中间checkpoint暂不删除。保留删除记录。 |
| 结果统计与分析 | 固定数据人口、指标/阈值、比较对象、置信区间及规则化诊断；统一输出结构化结果和模板报告。缺失或失败明确标记，不用模型补结论。 |
| 数据轨迹采集 | 固定任务/容器身份、采集字段、采集时机、去重键和存储位置；支持续采及来源追溯，不逐轮人工选择。 |
| 数据集采样 | 固定split、seed、规模、类别均衡及长度bucket规则；输出样本ID清单和数据版本，禁止跨split泄漏。 |
| 工程流程串接 | 配置化定义依赖、准入条件、资源锁、终态、超时和有限重试；复用共享入口，禁止重复派发，观察超时不当作训练失败或重启依据。 |

每次运行记录脚本/配置版本、输入身份、输出清单、状态及错误；提供幂等执行和单实例或
去重机制。普通训练监控仍为900秒。异常按配置记录并通知，不自动调用模型修复、扩大
删除范围或修改实验配置。固定脚本可按已授权的保留策略执行，无须每轮人工确认清单。

本节是执行要求，**不表示现有脚本已经全部归并完成**。现有监听与终点统计脚本继续
复用；一次性的逐step脚本应在后续独立工程变更中收敛为参数化入口，经有限验收再用于
新任务。不得原位修改当前运行或排队任务绑定的源码、配置及依赖，也不因此新增后台任务。

首个完成归并的入口为已提交checkpoint的TRAIN信号统计：
`tools.reports.committed_signal_report --config ...`。真实step512归档输出一致及幂等验收
已完成；具体命令、能力限制及剩余范围见[日常操作归并登记](routine-operations-v1.md)。

### MINDS 末点评测

当前简单跨数据集评测使用 MINDS native14。Stage1 和 Stage2 末点可以直接选用独立 HF 分片。先用同一检查点在目标机生成无标签 HF 对照面板，再冻结分片配置。每个单卡分片必须精确重现其所分配的对照记录，全量才能启动。SLURP 服务评测沿用数值一致性门槛。

从已有端点封包接续时，使用 `tools.reports.prepare_stage2_view_selection` 与
`configs/portable/operations/prepare-stage2-view-selection.json`。本地宏
`STAGE2_VIEW_SELECTION_PREPARE.hf_successor` 绑定先前封包、有限与全量配置、对照面板和
HF 原始输出的 SHA-256，以及新输出、目标机与设备。遇到已发生的服务对照失败可附上
`pins.comparison`，直接计划 HF 时只需同策略 HF 对照。用生成的 `operations-local.json`
交给 `tools.reports.native_seal_operations`。以 `run/audit-hf-shards-collected.json` 中的完整
分母、检查点和预测哈希确认结果。评测主报平均 Top‑1 原始概率、Top‑1 macro‑F1 和准确率。

Stage2 的 SLURP selection 可在新评测包的本地宏中用 `selection_steps` 指定已审核检查点的有序子集。
`selection_cache` 可绑定旧封存清单的路径及 SHA-256 与目标机只读缓存目录。封包时须确认两份封存的
录音清单、候选定义和评分标签逐字节一致。推理时按已验证的缓存收据核对张量与 processor 身份。

### FSC 历史评测

FSC 归入历史参考，从新评测队列移除。step 160 的官方 VALID 提示对照、封存配置与已审核结果见 [封存与评分方案](fsc-keyword-evaluation-v3.md)。关键词与整句示例均从官方 TRAIN 的书面文本标注按固定脚本提取。

## 已记录的状态快照

较新本地终态证据：原GRPO容器`e564df04`于`2026-11-15T12:51:47.17520631Z`
正常退出（ExitCode0、nonOOM）。原driver599307及collector1174133实际存活，阶段为
`collect-training`；终点统计969099仍等待原实验完成。原transport收据为
`transport_pending_not_audited`，含52007文件；本次归档快照21765513216字节，尚未封闭。
本次只读取原本地收据和准确进程句柄，没有远程进度查询或新分析任务。原收集→独立
训练审核→完整11点selection链继续，不能把正常退出视为完整审核成功或RL收益。
下方517/634等记录均为更早历史快照。

本次核对普通观察收据：GRPO local1794739236.6554358四rank517/634、running/nonOOM，
最近提交512；输入预处理已exit0，三组全量输入日程已发布并独立审核通过。
本地已核对原训练、后继队列及监听/分析进程存活；最近两次普通观察间隔均为900秒，
下一普通观察不早于1794740136.6554358。完整RL selection尚未开始。
这些是有时间戳的快照，最新状态以`.local/decision-objective-research/ordinary-observations-v1/`
中的`grpo-*-result.json`和`inputs-*-result.json`为准，不把文档数字当实时进度。

- 原训练及输入监听：`tools.reports.watch_context_existing_jobs`，每900秒，PID913576。
- 原队列后继监听：`tools.reports.watch_context_successor_jobs`，PID954637；覆盖RLOO、22个
  selection点及FSC/TEST/MINDS，等待原launch出现才开始计时。见[范围](context-successor-observer-v1.md)。
- 完整曲线后本地分析：PID969099等待原owner，重新审核并生成固定终点决策及NLL/Brier区间；
  见[衔接](context-endpoint-followup-v1.md)。目前尚无完整RL分析结果。

以上监听/分析不派发训练、不接管收集、不选新checkpoint，也不自动放行复杂训练。
**简单任务必须在完整selection上确认明确收益，才准入复杂训练**；训练reward上升、
TRAIN批次改善、输入处理完成或CPU检查通过均不满足此条件。

旧入口逐字归档于[状态整理前快照](archive/context-prob-v1-baseline-before-monitoring-consolidation-v1.md)，
更早的[arithmetic-v2整理前快照](archive/context-prob-v1-baseline-before-arithmetic-v2-consolidation.md)保留。
配置与运行包不可变；本文档更新不修改实验身份或拼接曲线。

## 学术与迁移交付边界

| 工作 | 现有证据 | 未完成项 |
| --- | --- | --- |
| FSC SpeechBrain官方权重复测 | 3793条，3774正确，99.4991%；FSC监督训练参考 | Jev的同人口迁移结果仍待原队列 |
| SLURP Omni clean原生意图参考 | 5207/12991，40.0816%；格式失败保留计错 | Jev clean TEST仍待冻结策略评测；非同预算比较 |
| SLURP HuBERT官方复测 | 固定接口与资产清单，作者数字仅文献 | 四项权重/依赖缺失788105608字节；当前不新增模型下载 |
| AudioJev官方路线 | 已有研究登记 | CUDA资源及实际运行证据未就绪 |
| 简单SLURP RL | Stage1完成；双信用GRPO为当前主线 | 完整两曲线、概率质量与不确定性、TEST收益 |
| FSC/MINDS跨数据集 | 完整输入准备；来源模型仅由SLURP selection选定 | 实际迁移推理及初始/选中策略比较 |
| 复杂Q任务（deferred） | 历史准备证据保留归档 | 留到下一阶段，本轮不调研、开发、训练或评测，不作为交付依赖 |

学术基线口径见[横向比较协议](../../experiments/academic-baseline-comparison-v1.md)、
[FSC比较](fsc-external-reference-comparison-v1.md)、[Omni clean参考](../research/slurp-omni-clean-reference-cluster-v1.md)、
[HuBERT资产缺口](../research/slurp-hubert-offline-asset-inventory-v1.md)。
当前输出独立事件概率，不输出开放槽值，不能用意图准确率填写完整SLU-F1。
MINDS8168条是上游TRAIN来源的迁移诊断，不能称官方TEST。扩展集登记不意味着全部已运行。

## 1. 当前状态和唯一配置入口

唯一配置目录为 `configs/releases/context-prob-v1/`：`experiment.json` 定义任务和实验，
`algorithms.json` 登记算法，`LOCK.json` 固定源码、镜像、模型、数据和日程。
lock ID 为 `2ece49811d1e11eabc75f3e95e73ef532bda3fe9f16e9a399da7725ad0d95bfc`。
本次重新校验364项通过。配置中的 `stage2_released=false` 等字段是历史快照；
后续发布由独立准入证据与运行包说明，不修改历史JSON来更新状态。

当前数值协议为 `context-bernoulli-probability-cpu-log-v2`，正式GRPO发布ID
`493cc1bea2fafc93a18827707f11ee9cf5308ee7078990c1e52d9d425a422d49`。
修复统一actor/rollout的概率到log计算路径，未放宽误差门槛；真实DP4连续更新、
新进程恢复及四rank master/Adam/采样/执行/joint权重比较均已通过。
完整准入见 [数值版本验收](context-arithmetic-formal-acceptance-v2.json)。
原首批失败属于数值表示路径不统一，没有OOM或镜像损坏证据；旧现场留档。

| 工作 | 当前证据与结论边界 |
| --- | --- |
| Stage1 | 完成634更新，终点完整9226选择集：set-F1 73.8041%、top1 82.0724%、macro-F1 77.4315% |
| 概率质量 | 同一终点NLL 0.0187434286、Brier 0.0046499592；不是TEST或RL成绩 |
| 训推一致性 | 终点256条HF/vLLM面板概率差0；修复RL连续/恢复核验通过，不代表任意输入均已验证 |
| 正式GRPO | 原容器 e564df04 已正常退出；完整 11 点 selection 未收口，以收据为准 |
| RLOO-clipped | 已停止作为当前运行分支（2026-11-17 协作指南）；404 文件包保留为历史，不再派发 |
| FSC迁移 | 条件队列等待双曲线完成；3793条目标输入已准备，没有迁移成绩 |
| SLURP clean TEST | 12991条输入及processor缓存已验；等待冻结来源pair，没有实际TEST模型成绩 |
| MINDS迁移准备 | 8168条16k输入/processor/独立缓存审核通过，固定镜像76项CPU检查通过；driver644727条件等待clean TEST完成后接续，尚无NPU对齐或迁移成绩，不阻断原队列 |
| 评测工具 | 固定镜像无NPU 110 tests通过/30.63秒，已独立收集；相关模型/来源/SSH夹具使用替身 |

step64、256、320、384、448、512已提交检查点均有独立CPU归档审核；step512的四卡FP32 master重组转BF16，
与全部128参数/6,878,209元素执行/发布精确一致。这不是新增NPU恢复或完整训练效果。
见[第512步诊断](../research/context-step512-training-review-v1.md)。

Stage1适配前后set-F1为3.3191%→73.8041%，top1为1.7993%→82.0724%；
起点是新初始化概率头，不能称原生ASR零样本。Stage2 step0使用完成Stage1的终点。
TRAIN阶段诊断显示集合F1改善、NLL/Brier恶化与大量恒奖励组共存；完整selection结论仍待。

## 2. 先固定简单任务和结论边界

输入为原始音频、完整候选定义和统一任务规约；Qwen3-ASR-0.6B加共享浅Transformer head，
一次前向输出60个独立sigmoid概率。候选定义随任务配置变化，head不使用固定类别专属参数。
任务`exclusive_event`判断“该候选是否为benchmark指定的目标事件”。
SLURP单标签不提供完整多意图存在性真值；不据此构造语义存在性负例或槽位缺失负例。

harness按冻结规则消费概率：主决策为阈值0.5的候选集合，另报top1以连接原生意图基线。
独立sigmoid允许空集和多选；这与强制单选softmax不是同一动作空间，成绩不能互换。
不生成文本CoT，不走ASR转写后再调用语言模型，不在当前阶段生成开放槽值。

主指标为query macro set-F1@0.5；同时报告top1 accuracy/macro-F1、精确集合、
集合precision/recall、返回数量、空/多选率及NLL/Brier。奖励、排序和概率质量分别判断；
sigmoid和排序loss都不自动保证校准。选择规则固定为set-F1最高、NLL最低、最早步。

## 3. 一套共享工程，三种用途配置

| 项目 | 本轮固定值 |
| --- | --- |
| 镜像 | `sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c` |
| 模型 | 本地`/mnt/cephfs/xingchao/models/Qwen__Qwen3-ASR-0.6B-hf` |
| 参数适配 | q/v LoRA，rank4、alpha8、dropout0；共享一层Transformer概率头 |
| head | hidden1024、8 heads、FFN1024、residual_scale0.1、dropout0 |
| 精度 | BF16执行和导出；FP32 master、Adam和loss |
| 并行 | 175卡4–7，verl FSDP数据并行DP4，vLLM TP1；沿用原设备租约 |
| batch | global128 = 4 ranks × micro8 × accumulation4 |
| 优化器 | AdamW，lr1e-4，betas(0.9,0.999)，weight_decay0，grad_clip1 |
| 数据日程 | 全量覆盖、类别分散、global batch内部长度分组；补齐样本权重0 |
| 预算 | Stage1两轮634更新；每个Stage2算法两轮634更新 |
| runtime | `qwen3-asr-context-bf16-packed-row-v1` |

三种用途为固定策略评测、Stage1指令/候选绑定适配、Stage2任务RL；共享processor、
音频管线、head、actor、rollout、保存恢复、评测与日志，不按算法复制训练工程。
Stage1是在SLURP协议上适配消费能力，尚不能称跨任务通用instruction能力已经学会。

热状态写入`/data`，Ceph共享模型/数据只读；保留本地fsync，遵守
[runtime-storage-policy.md](runtime-storage-policy.md)。普通远程观察间隔900秒。
观察失败或超时不等于训练失败，不据此重建任务；原设备锁和准确容器句柄继续有效。

## 4. 当前RL算法和冻结损失

实际调用：`span_group_advantage.py`使用verl的GRPO/RLOO优势估计器，
`span_sampled_hybrid.py`使用`compute_policy_loss_vanilla`，
`context_native_actor.py`复用`DataParallelPPOActor`更新循环。
复用PPO类和clipped surrogate不等于启用了critic-PPO。

| ID | 定位 | 优势估计 | 当前可证明范围 |
| --- | --- | --- | --- |
| `grpo_v1` | 正式主线 | `(R_i-mean(R))/(std(R)+epsilon)` | 修复连续/恢复通过，正式新版运行中，完整曲线待 |
| `rloo_v1` | 同预算对照，报告名RLOO-clipped | `R_i-mean(R_j,j!=i)`，不除组标准差 | 实际单步更新证据；新版全量包封存，等待GRPO成功后派发 |
| `ppo_critic_design_v1` | 设计预留 | learned value baseline | value训练/恢复未实现，不可派发 |

每条真实query一次前向取得概率，再采样G16个完整60-bit动作；G16不是16倍音频前向。
奖励为完整候选集合F1，不能用59个负例主导的逐位准确率替代。
联合log概率为各Bernoulli log概率之和，联合动作ratio进入clip=0.2、dual-clip=3的surrogate。
每行为批次一个更新epoch，先算完整组优势再划分DP，零信号组保留在原曝光分母中。

Stage1：`L = L_rank + L_binary_NLL`。
Stage2：`L = L_RL + 0.01 KL(pi_theta || pi_Stage1) + alpha(t)L_rank + alpha(t)L_binary_NLL`。
KL逐候选求和、每query计一次；alpha从0.3在前80%更新中余弦衰减到0.1。
两个算法共用完整Stage1 step634起点及固定reference，进入Stage2重置optimizer。
数据顺序、曝光、G16、优化器、辅助系数、评测点均一致；另外报告真实NPU成本和优势尺度。

RLOO在固定G下等于`G/(G-1)*(R_i-mean(R))`；本对照主要识别标准差归一化和优势尺度影响，
不能称两种不同推理机制。尺度变化也会改变RL相对KL/辅助loss的作用。
两者在组内奖励全部相同时，RL优势均为零；更换估计器本身不能解决此类无信号组。
每个行为批次仅一次更新，更新前ratio约为1；clip参数不保证这一次optimizer移动受到
信赖域约束，必须结合更新后KL、概率质量及完整选择集曲线评估稳定性。
两者都是概率动作适配；不是原论文自回归token配方逐项复现，RLOO-clipped也不是未改动的REINFORCE。
本轮检验混合RL方案相对Stage2 step0的持续改善，不新增RL-vs-SFT比赛，也不声称已隔离纯RL因果贡献。

## 5. Step-wise先设计信用与监督，不先改标准算法

Q1/Span1、Q2/Span2、Final/Span3是输入预设、一次前向的并行判决；问题编号不是环境时间。
当前Final不会读取同层刚输出的其他概率，所以不沿Q编号套temporal GAE或token reward-to-go。
如果未来要求前置prob参与Final，必须先定义内部依赖或harness聚合，再冻结新工程；不能口头假定已有。

复杂任务仅在Q的可迁移语义、合法动作、过程真值及缺标mask均固定后开展三组比较：

1. Direct：直接Final判决。
2. Q-input：增加统一语义问题，只使用Final reward。
3. Q-credit：与Q-input相同模板、动作、Final辅助loss及更新预算，仅增加可靠过程reward。

为避免混淆，Direct与Q-input均关闭过程辅助loss，只保留相同Final辅助loss；
Q-credit同样不另加过程辅助loss，因此Q-input与Q-credit只差过程reward。
若要研究过程监督，另设Q-supervised：与Q-input同输入和Final reward，仅增加过程辅助loss。
该组不是当前简单基线的前置任务。新增Q会改变输入长度，除相同训练曝光/更新预算外，
必须报告实际NPU时间与Final推理成本，不能直接声称计算预算相同。

Q不得依据当前query答案或gold选题；缺标mask，不补false。Final标签的确定性投影明确称属性辅助监督。
槽位过程监督最新工作包为current-rules-v14，全部40513录音绑定已重验；161当前明确参考、
10未决、4803旧规约待重审、500586未评，正式语义准入仍为0。类型v16/政策v16/边界v17
不能与早期v1/v6混用；既有教师三值校准未通过，不能重复派发或当作完整过程gold。
见[当前槽位监督支持范围](../research/slurp-slot-workpack-learning-support-v14.md)。这不影响当前简单事件任务。
第一版仍用原GRPO/RLOO估计器：`R = R_final + lambda R_process`；先冻结过程任务族、
权重及有效标签归一化规则，使多问几个问题不自动增加reward。过程指标改善不能代替Final改善。

**需重点验证GRPO的尺度问题：**若组内Final奖励恒定，过程奖励有差异且epsilon可忽略，
`normalize(R_final + lambda R_process) = normalize(R_process)`（lambda>0）。
因此lambda很小也不保证过程梯度小。需报告Final恒定组、奖励方差、有效优势组、clip fraction、
KL、概率饱和、过程/Final指标和成本；这些是后续诊断要求，不声称当前埋点全部具备。

只有实测显示信用分配受阻，才另登记分项优势或节点反事实信用变体；二元节点在标签可评分时
可以枚举替代动作，无需额外音频前向。节点ratio/clip不等于联合ratio/clip，须独立命名、验证和比较。
不把复杂任务、节点信用或PPO作为当前简单RL发布前提。

## 6. 有限收口顺序与变更规则

1. 继续已发布的GRPO：634步训练、独立收集/审核、11点完整9226选择集评测。
2. 从同一个Stage1终点重置optimizer，运行同预算RLOO-clipped及相同11点评测；报告全曲线与退化。
3. 按预先冻结的SLURP selection规则选择GRPO checkpoint；FSC仅改定义和模板，禁止目标集调参。
   随后用同一来源pair执行clean TEST，目标标签仅用于预先声明的评分，不参与选模。
4. 简单任务完整selection确认明确训练收益后，才将有可靠监督的复杂Q设计转为新实验。
   训练reward上升或训练完成不足以放行；收益不明确时继续简单任务诊断及独立改进。
   PPO、复杂Q、micro32调优、
   新学术模型下载均不成为当前正式实验的额外前置门槛。

评测点固定为 `0,64,128,192,256,320,384,448,512,576,634`。
本轮检验混合RL方案相对Stage2 step0的变化，不承诺曲线单调，也不把辅助监督的效果全部归因于RL。

只维护一套共享工程。任务、reward、loss、模板、batch、精度或数据日程变化需要新实验身份，
不能修改运行包或拼接前后曲线。仅处理真实阻断缺陷，有限验收只覆盖受影响链路；
成功的Stage1、缓存、checkpoint和评测产物复用。超时不是终态，不重启已有准确句柄。
能重收集解决的问题不重训。新增吞吐优化单列候选，不打断正在运行的基线。

三类基线分别验收：文件身份固定、真实NPU工程可靠、学术指标改善。前两类不替代第三类。
本次只整理入口与核验既有身份，没有修改算法、运行配置或排队依赖。

## 7. 固定工程的变更边界

本轮不是重新搭建三套框架。固定策略评测、Stage1、Stage2共用一套runtime；
GRPO和RLOO-clipped通过配置切换。PPO-critic只登记设计状态，不把尚未实现的value链路称为稳定配置。
本页是状态入口，原配置JSON是不可变设计快照，正式包与验收收据是当前发布身份；三者用途不同。

| 变化 | 处理办法 | 是否阻断当前简单基线 |
| --- | --- | --- |
| 文档、独立结果分析 | 写入未被运行包绑定的报告，保留原始证据 | 否 |
| 新任务、Q、过程标签、reward、loss | 先完成任务协议及比较设计，再封存新配置 | 不影响原运行，只约束新实验 |
| microbatch、global batch、分桶、数值实现 | 独立版本验证受影响的训推和恢复路径；新曲线独立身份 | 否，不原位改当前包 |
| 真正阻断训练/评测的缺陷 | 保留失败现场，只修受影响候选链路，复用已经通过的证据 | 仅阻断受影响运行 |
| SSH观察超时、日志暂未更新 | 保留原容器和租约，按900秒节奏或准确wait核实 | 不自动视为失败 |

有限工程验收只涵盖实际输入、训推概率、连续更新、策略刷新、保存恢复及评测收集。
已经通过的验收不因研究思路变化而反复重跑；完整训练和完整曲线才用于证明全程可靠性及效果。
复杂Q、PPO、吞吐候选和新数据集不追加为当前简单任务的发布门槛。

本次再次执行冻结清单校验，364项身份全部通过；没有修改冻结源码、实验参数或排队包。
独立整批IPC候选在相同镜像CPU通过32项检查，但尚无真实NPU收益，仍不升级为正式基线。
其验收范围见[整批IPC候选](context-bulk-ipc-candidate-v1.md)。

任务设计的最小交付是：输入/候选语义、输出prob及harness规则、可观测gold与缺标mask、
动作/reward/loss、数据划分、主次指标和选模规则。以上固定后才冻结工程配置。
对复杂Q还须说明Final如何利用前置信息：当前共享上下文及因果隐藏状态可以提供信息，
但同层读出的前置prob没有再次输入Final。显式prob依赖需要新的结构或聚合规则。
不因加入多个Q就宣称实现了多步决策或获得过程信用收益。

停止扩建的标准是上述协议冻结且受影响的有限工程验收通过，即进入既定训练和评测；
不能把新增数据集、更多算法或更多诊断变成已准入实验的继续条件。

当前效率证据已跨入第二轮：首轮末16步均102.16秒，第二轮前14步均68.22秒，
reference miss2048→0，actor约24.4秒不变。两个不同TRAIN窗口的步内观察，不能称新代码
加速或整轮wall吞吐；见[连续窗口分析](context-reference-epoch-windows-v1.md)。
下一版本优先验收行为策略rollout批量化，再判断microbatch；1090文件有限NPU probe已封存，
仍须原队列释放资源及设备租约，未派发。见[有限验收包](context-batch-probe-prepared-v1.md)。

分项日志候选使用真实verl在固定镜像CPU通过16项检查，loss/梯度/Adam一致；尚非完整
分布式actor或Qwen/NPU验收，未安装到本次训练。见[验收说明](context-loss-metrics-native-validation-v2.md)。

全量复杂候选输入及三组日程已完成v2审核发布：40513录音、81026请求；
13文件独立核对、每轮全覆盖和三组曝光/DP slots一致。见[完整发布](context-attribute-full-input-publication-v2.md)。
Direct平均2011.39 token、Q-input平均3048.39，每条固定增加1037 token；总量增加51.56%。
输入长度平方和比2.29684仅为attention几何代理，不是实际模型FLOPs、HBM或训练时间。
后续比较须报告真实成本；输入就绪不放行复杂训练。见[完整长度统计](context-attribute-full-input-geometry-v2.json)。

## 阶段结果如何驱动下一轮简单任务优化

阶段TRAIN诊断用于发现问题，正式效果判断使用完整11点selection曲线及固定终点配对区间。
最高分checkpoint依照既定规则用于后续评测；挑选后的最高分不能直接当作预先固定终点的
显著性证据。完整曲线同时展示末期退化，TEST不用于决定下一组loss/reward参数。

| 完整selection上的现象 | 下一轮优先研究 | 比较约束 |
| --- | --- | --- |
| set-F1上升但NLL/Brier恶化 | 概率辅助项衰减或KL强度，先选择一个变量 | 同起点、人口、曝光和harness；同时报告决策效用与概率质量 |
| set-F1无明确改善，TRAIN错误组大量恒reward | 原RLOO对照后，评估既有条件积分或精确单正例F1候选 | 估计器/梯度尺度独立命名；仍是简单任务，不以复杂Q绕过收益要求 |
| top1与set-F1方向不同 | 分解空集、多选、阈值错误与排序错误 | 主指标和0.5阈值保持；不事后改为只报表现较好的指标 |
| actor时间稳定，准备/rollout占主要开销 | 优先验收已封存的批量rollout候选 | 同输入、策略、概率和采样；同时报告实际耗时、峰值内存及成本 |

以上是条件分支，不提前指定获胜算法或许诺收益。精确积分减少有限采样误差，仍优化集合
效用，不能保证校准或消除sigmoid饱和。binary NLL/Brier用于识别概率质量问题；若将其
改成reward，目标和尺度也随之改变，应另立实验，不能称为同一F1任务的纯方差优化。
完整结果未支持复杂任务准入时，继续简单任务的独立改进；无需先运行复杂Q来验证这些候选。
新增日志候选与吞吐候选须完成各自受影响的有限NPU验收，当前运行包和后继队列保持不变。

## 方法依据

- GRPO：DeepSeekMath，https://arxiv.org/abs/2402.03300 ，§4.1.2/4.1.3。
- RLOO：Back to Basics，https://arxiv.org/abs/2402.14740 ，§2.3。
- PPO：https://arxiv.org/abs/1707.06347 。

此前已核对原文，本地来源与hash见
`.local/decision-objective-research/baseline-method-sources-20261115/receipt.json`。
方法说明不提供本项目效果证据；运行实现以固定源码hash为准，不随上游latest升级。
更完整方法论证见 [研究执行决议](../research/context-baseline-execution-decision-20261114.md)，
其中历史状态不覆盖本页与独立终态。

