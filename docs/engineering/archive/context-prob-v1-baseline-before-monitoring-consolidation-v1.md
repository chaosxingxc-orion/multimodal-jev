# context-prob-v1：当前任务、工程与算法基线

本页为当前唯一执行入口。执行顺序固定：**任务与比较设计 → 工程及配置冻结 → 有限验收 → 正式训练**。
最新自动观察：GRPO local1794724836.561848四rank308/634、running/nonOOM，最近提交256；
下一>=1794725736.561848。TRAIN inputs local1794724654.8001697为31232/40513、running/nonOOM，
下一>=1794725554.8001697。两任务定时器首轮实际观察均与前次相隔900秒以上，见
[自动观察收据](../context-background-observer-first-cycle-v1.json)；以下较早进度均为历史。
普通观察由`tools.reports.watch_context_existing_jobs`后台定时器按900秒驱动原observer；
单实例锁和共享attempt避免手动/自动重复查询。仅监听原GRPO与输入处理至各自终态，
不接管原收集队列，不自动修改配置或准入复杂训练；科研分析与后继RLOO监听另行接续。
运行状态见`.local/decision-objective-research/ordinary-observations-v1/watcher-status.json`。
最新普通观察以统一observer收据为准：local1794723936.5562925，原GRPO四rank299/634、
running/nonOOM，已提交至256；下一普通>=1794724836.5562925。
TRAIN输入local1794723754.7960172完成29696/40513，running/nonOOM，下一>=1794724654.7960172。
当前步总100.65秒，其中准备74.22秒、actor23.46秒；仅耗时观察，没有新效果评测。
下列live-v编号和到期时间为历史记录。下一阶段诊断关注第二epoch参考缓存与step320提交，
不修改原曲线，不以TRAIN单批收益放行复杂训练。
最新普通live-v45：local1794722078.7953887，原GRPO四rank281/634、running/nonOOM；
下一>=1794722978.7953887。TRAIN下一>=1794722766.4990485。下方更早到期时间仅历史记录。
完整曲线后可使用[固定终点配对分析](../../research/context-simple-endpoint-uncertainty-v1.md)，
该入口未派发、不会自动放行复杂训练；真实完整成功路径仍待。
最新用户要求：普通观察每900秒；简单任务在完整selection确认明确训练收益后才允许复杂训练。
GRPO live-v44四rank272/634，下一普通>=1794722060.694339；TRAIN下一>=1794721860.9853616。
以下历史观察收据中的600秒到期值不再作为新的调度依据。
最新普通观察live-v43：local1794720525.3656836，原GRPO容器running/nonOOM，四rank266/634；
下一普通不早于1794721125.3656836。364项冻结身份本次重验通过，运行配置不变。
本次[收口决议及step-wise信用边界](../../research/context-stable-baseline-and-credit-decision-v1.md)
补充独立输出/加性奖励的梯度边界；下方live-v42及更早观察保留为历史证据。
配置、运行状态、效果证据分别记录；运行包不原位修改。此前入口逐字保存在
[历史入口](context-prob-v1-baseline-before-arithmetic-v2-consolidation.md)。

本轮收口决议：复用已发布 arithmetic-v2 完成简单任务，不继续扩建当前基座。
算法配置锁定 GRPO 主线、RLOO-clipped 对照；PPO-critic 保留设计身份，不能计入已验收基线。
新复杂任务先冻结问题、标签来源、动作空间、奖励和比较，再接共享工程并有限验收。
当前阶段的交付是完整训练/选择集曲线与冻结策略评测，不以新增工具数量代替交付。

最新普通观察live-v42（local1794719863.34297）：原GRPO容器仍running/nonOOM，
四rank259/634，256步checkpoint提交，下次不早于1794720463.34297。
TRAIN输入processor最近同容器24064/40513（live-v21，local1794720315.546464），
仍running/nonOOM，待终态后独立审核；下一普通观察不早于1794720915.546464。
下一版本吞吐候选已补分段attention/audio、scheduler桥接及批量encode owner。
104项固定镜像CPU检查通过；随后真实Ascend metadata导入在无NPU容器触发硬件查询失败，
保留exit1现场，不称整模型接口/吞吐验收，不改正式基线。详见
[批量owner候选与限制](raw/context-rl-history/context-batched-owner-candidate-v1.md)。
独立批量服务已复用原bulk协议，通过固定镜像28项CPU socket/分组/清理检查；真实128条
输入已完成容量分组计划，无NPU吞吐证据。见[服务候选](raw/context-rl-history/context-batched-service-candidate-v1.md)。
有限serial/batch4 NPU包已封存1090文件，真实Stage1策略和TRAIN128，未派发；计量记录实际
语言forward与worker峰值内存，尚无加速结论。见[验收封包](../context-batch-probe-prepared-v1.md)。
181固定镜像复制已完成，原始配置文件SHA256精确等于175镜像ID，33层身份一致。
原finalizer因Docker inspect额外默认字段而失败的记录保留，独立v2身份核验通过；
不等于181 NPU运行验收或设备释放。见[镜像身份收据](../context-fixed-image-replication-181-v2-verified.json)。
有限批量probe派发入口5项本地检查已通过，实际1090文件封包再次校验通过；仍未派发。
181 Omni B此前已确认使用4–7卡，不能把进程空窗当作资源释放。
收集后独立审核入口已补齐，11项本地检查通过，覆盖输入/G16重建、终态及完整传输身份、
四轮精确比较和实际前向长度/计时；概率与设备telemetry为替身，尚无NPU加速证据。
见[审核证据](../context-batch-probe-independent-audit-v1.json)。

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
完整准入见 [数值版本验收](../context-arithmetic-formal-acceptance-v2.json)。
原首批失败属于数值表示路径不统一，没有OOM或镜像损坏证据；旧现场留档。

| 工作 | 当前证据与结论边界 |
| --- | --- |
| Stage1 | 完成634更新，终点完整9226选择集：set-F1 73.8041%、top1 82.0724%、macro-F1 77.4315% |
| 概率质量 | 同一终点NLL 0.0187434286、Brier 0.0046499592；不是TEST或RL成绩 |
| 训推一致性 | 终点256条HF/vLLM面板概率差0；修复RL连续/恢复核验通过，不代表任意输入均已验证 |
| 正式GRPO | 最近已归档普通观察local1794719863.34297：容器e564df04f946f853 running/非OOM，四rank step259/634；不是完整训练结果 |
| RLOO-clipped | 同数值版本404文件包已封存，源码/initial与GRPO相同；等待GRPO完整训练和11点选择集成功 |
| FSC迁移 | 条件队列等待双曲线完成；3793条目标输入已准备，没有迁移成绩 |
| SLURP clean TEST | 12991条输入及processor缓存已验；等待冻结来源pair，没有实际TEST模型成绩 |
| MINDS迁移准备 | 8168条16k输入/processor/独立缓存审核通过，固定镜像76项CPU检查通过；driver644727条件等待clean TEST完成后接续，尚无NPU对齐或迁移成绩，不阻断原队列 |
| 评测工具 | 固定镜像无NPU 110 tests通过/30.63秒，已独立收集；相关模型/来源/SSH夹具使用替身 |

此前本地检查driver599307/599537/599824/600506均存活；历史进程存活不等于当前终态。
下一次GRPO普通远程观察不早于local1794720463.34297；不为状态答复加密轮询。
证据分别为 `context-arithmetic-formal-live-v42.json`、
`context-stage1-selection-step0634-v2.json`、`slurp-context-test-cpu-v1.json`。
完整RL提升、TEST表现及跨数据集迁移收益均未取得，不能提前宣布基座已通过全程稳定性验证。
step256提交也已独立收集并通过同范围审核：全部128参数/6,878,209元素由四卡FP32 master
重组转BF16后与执行/发布精确一致，Adam、adapter、marker及激活记录匹配。
这是CPU产物审核，不是新NPU恢复或效果。见[第256步审核](raw/context-rl-history/context-arithmetic-step256-independent-v1.md)。

step64提交已独立收集并核验四卡FP32 master/Adam、BF16执行与发布策略、native adapter、
冻结marker及四rank激活记录；全部128参数/6,878,209元素从四卡master重组转BF16后
与execution逐元素一致。见[检查点审核](../context-arithmetic-step64-independent-v1.json)和
[master与导出重组核验](../context-arithmetic-step64-master-export-v1.json)。这不是新NPU恢复、
完整训练审核或选择集成绩。以下动作诊断仍来自live-v13第74步。
同次记录的128输入显示G16动作全同概率均值83.32%，固定reference为10.07%；
这是无奖励差异概率的下界，不能区分正确/错误确定性，也不是实际采样计数。
见[动作多样性诊断](../../research/context-live-action-diversity-step74-v1.md)，运行配置保持。
同次TRAIN真值绑定后，集合正确86→98/128，top1 104→101/128，NLL/Brier变差；
20条集合错误输入的G16奖励全同概率>95%。这是单批诊断，不是泛化或完整曲线。
详见[奖励信号与错误确定性](../../research/context-live-reward-support-step74-v1.md)。

同批真实采样重放已完成：128组动作hash全部匹配，86组全1、20组全0、6组恒部分奖励，
只有16组（12.5%）奖励有差异；30条阈值决策错误中26条恒奖励。这是实际采样计数，
不同于前面的分布期望；独立绑定TRAIN/audio/tensor/gold，四rank副本只计一次。
5项本地检查通过，优势与输出logit方向使用本地公式，不是原生verl/参数梯度重放。
见[实际奖励组](../../research/context-actual-reward-groups-step74-v1.md)。不据单批改变运行配置。
后续固定镜像真实verl复核完成：GRPO/RLOO均16/128组非零优势、112恒奖励组精确零；
与本地公式最大差0/7.8231e-8。容器exit0/nonOOM，无NPU；这不覆盖actor参数梯度。
见[原生优势复核](../../research/context-reward-replay-native-verl-v1.md)。

当前Stage1适配前后set-F1为3.3191%→73.8041%，top1为1.7993%→82.0724%；
起点是新初始化的概率头，不能称原生ASR零样本。Stage2的step0使用已完成Stage1的终点。

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
[runtime-storage-policy.md](../runtime-storage-policy.md)。普通远程观察间隔900秒。
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
见[当前槽位监督支持范围](../../research/slurp-slot-workpack-learning-support-v14.md)。这不影响当前简单事件任务。
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
其验收范围见[整批IPC候选](../context-bulk-ipc-candidate-v1.md)。

任务设计的最小交付是：输入/候选语义、输出prob及harness规则、可观测gold与缺标mask、
动作/reward/loss、数据划分、主次指标和选模规则。以上固定后才冻结工程配置。
对复杂Q还须说明Final如何利用前置信息：当前共享上下文及因果隐藏状态可以提供信息，
但同层读出的前置prob没有再次输入Final。显式prob依赖需要新的结构或聚合规则。
不因加入多个Q就宣称实现了多步决策或获得过程信用收益。

停止扩建的标准是上述协议冻结且受影响的有限工程验收通过，即进入既定训练和评测；
不能把新增数据集、更多算法或更多诊断变成已准入实验的继续条件。

当前效率优先级也有实测依据：正式步骤2–29的rank0记录显示平均总耗时100.43秒，
rollout65.67秒，其中固定reference32.61秒；actor23.95秒，输入缓存3584/3584命中。
下一版本先评估串行rollout服务批量化和固定reference复用，再评估micro32。
这只是前缀计时，不是最终独立训练审核或已经实现的加速；详见
[正式RL吞吐分析](raw/context-rl-history/context-rl-throughput-prefix-v1.md)。不修改正在运行或排队的包。

## 方法依据

- GRPO：DeepSeekMath，https://arxiv.org/abs/2402.03300 ，§4.1.2/4.1.3。
- RLOO：Back to Basics，https://arxiv.org/abs/2402.14740 ，§2.3。
- PPO：https://arxiv.org/abs/1707.06347 。

此前已核对原文，本地来源与hash见
`.local/decision-objective-research/baseline-method-sources-20261115/receipt.json`。
方法说明不提供本项目效果证据；运行实现以固定源码hash为准，不随上游latest升级。
更完整方法论证见 [研究执行决议](../../research/context-baseline-execution-decision-20261114.md)，
其中历史状态不覆盖本页与独立终态。


## 历史执行收口（2026-11-15 09:40 北京时间，进度已由页首覆盖）

本轮只刷新状态与执行边界，未改算法、运行包或队列。364项基线身份及正式GRPO
1078项驱动依赖重新核验无变化；四rank131/634，64和128步checkpoint已提交。
这不替代后续独立终态审核或11点选择集曲线，当前仍没有完整RL收益结论。

当前稳定性应分层表述：发布身份已冻结、有限NPU训推/连续更新/恢复验收已通过，
完整两算法训练及下游评测尚未闭环。用户要求的多个稳定基线落实为同一工程的算法配置，
不复制框架；GRPO正式执行，RLOO-clipped为同预算队列，PPO-critic仅设计。

新任务必须先交付任务协议：部署时可构造的Q、候选语义、prob与harness消费规则、
gold来源/缺标mask、reward/loss、数据划分及主指标/选模规则。完成后才封存该任务配置，
进行受影响范围的有限验收，再训练；不要求当前简单任务等待所有复杂任务设计结束。

原生属性首轮比较已在
`configs/experiments/context-native-attribute-comparison-design-v1.json`前瞻固定：
Direct为60节点，Q-input/Q-credit均124节点（18 scenario、46 action、60 Final）。
Q-credit奖励为`Final_F1 + 0.25*(scenario_F1+action_F1)/2`；两Q组只差过程reward，
都不加过程辅助loss，同Stage1起点、G16、global128、634更新及11点选择集。
其标签来自Final确定性投影，不能称独立语义过程gold；仍不可派发，CPU objective通过
不等于完整actor或真实NPU已验收。Direct与Q输入/动作/KL支持不同，应按整套方法比较。

step-wise首轮保留标准GRPO/RLOO优势估计器和联合动作surrogate。重点验证：
Final恒定组是否获得有效信号、过程收益是否传递到Final、GRPO标准化是否放大过程项，
以及概率校准和单位NPU时收益。只有这些证据显示需要调整，才登记分项优势或节点
反事实信用的新变体；不能将多个并行Q直接改写为时间GAE，也不能悄悄替换联合clip。
前置prob未回灌Final，当前只能主张共享上下文/隐藏表示的利用，不能预设显式概率链成立。

交付顺序保持GRPO全曲线→RLOO-clipped同预算全曲线→来源选择后的FSC/clean TEST/MINDS。
复杂Q设计和独立候选验证可继续，但不改变上述任务的准入门槛。出现真实阻断缺陷时
仅修受影响版本；batch、loss、reward等研究调整另开实验身份，不拼接原曲线。


## 本次用户要求的执行决议（依据live-v41）

当前不重搭工程、不改变运行及排队配置。364项原始锁身份本次重新核验通过；
其stage2_released=false属于初版快照，后续正式发布与arithmetic-v2准入收据另行绑定。
文件锁定、有限NPU验收、完整学术实验三个阶段不能混称“全部稳定”。

| 阶段 | 固定交付 | 进入下一阶段的条件 |
| --- | --- | --- |
| 任务定义 | Q/候选语义、输出prob与harness规则、gold/mask、动作/reward/loss、划分/指标/选模 | 部署时能构造Q，标签支持目标，比较因素明确 |
| 工程锁定 | 同一runtime的评测/Stage1/Stage2配置；镜像、源码、processor、日程、算法身份 | 实际输入下训推概率、连续更新、刷新、独立恢复及评测收集有限验收通过 |
| 正式实验 | 全量TRAIN、固定11点selection、冻结策略TEST及配置迁移 | 报告完整曲线、退化、概率质量和成本；不能只给最佳点 |

简单任务已进入第三阶段，GRPO实际253/634；RLOO-clipped沿原条件队列接续。
复杂属性任务仍是前瞻设计及候选工程，不能用简单任务验收代替其124节点验收。

step-wise不立即修改原生估计器。先比较Direct、Q-input、Q-credit，保留标准算法对照。
若完整TRAIN/selection诊断仍显示有效优势稀疏、过程主导或联合ratio信用混淆，
才另立算法候选；第74步112/128恒奖励组是定位线索，不足以宣称全程规律。

可优先研究的两个调整方向（本段仅设计，不改变现有lambda=.25的属性比较）：

1. 分项信用：分别构造Final与过程优势，再按固定系数组合，避免先混合再组标准化
   让过程项在Final恒定组中抵消lambda。各项的尺度、零方差处理、分母和clip必须明确定义；
   这是一种新算法变体，不能沿用原GRPO名称后悄悄换目标。
2. 节点反事实信用：在已采样其他位不变时，以可验证reward计算
   `b_j = sum_b pi_old(a_j=b|x) R(a_-j,b)`，节点优势`A_j=R(a)-b_j`。
   二元动作可枚举0/1，不额外运行音频主干；节点baseline与行为概率停止梯度。
   只有完整可评分真值才可使用。节点ratio/clip与当前联合动作ratio/clip不同，
   必须独立命名并验证；不能保证消除概率饱和或用过程分数代替Final提升。

固定G时RLOO只是组中心奖励乘G/(G-1)，没有GRPO的组标准差归一化。
因此RLOO提供有用的尺度对照，但与GRPO一样不能从全等采样奖励中产生优势。
当前一步采样→一次更新的配置也不应只凭clip=.2宣称更新幅度受限；须看更新后KL。
PPO-critic暂不扩建：当前单步决策可用V(x)基线，Q序号仍不能充当时间步套GAE。
先由完整对照确定现有估计器是否不足，再评估value链路的新增成本。


第74步信用诊断补充：已有TRAIN128重新认证与G16 hash重建通过，CPU FP64计算单标签
集合F1精确期望梯度及节点条件积分方向。112恒奖励组含26阈值错误，均有非零方向；
最小正确候选方向约1.6e-6，仍有饱和限制。7本地穷举/梯度检查通过，未模型反向、更新或
heldout评测。详见[条件积分信用诊断](../../research/context-counterfactual-credit-step74-v1.md)。
此结果支持后续独立算法设计，不修改当前标准估计器或实验顺序。
