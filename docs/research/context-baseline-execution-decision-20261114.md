# 简单概率决策任务：工程收口与 RL 实验顺序

本页落实用户关于“先定义任务，再锁定工程，最后训练”的要求。执行配置仍以
`configs/releases/context-prob-v1/` 和 `docs/engineering/context-prob-v1-baseline.md`
为准；本页不修改运行中源码、配置或已排队验收包，不新增训练器。

## 当前执行约定（2026-11-15）

最新覆盖：修复版本独立恢复已通过，四rank master/Adam/采样/执行/joint策略精确；
原证据与数值补丁的实际连续/恢复链已经重新审核。正式GRPO新版已封存并启动，
容器e564df04f946f853，remote1794693401.3869965确认running/nonOOM，下次普通
观察>=1794694001.3869965。driver599307执行634步→收集→完整训练审核→11点评测。
仍无完整RL效果；不能把两步验收或running状态当收益。详见当前工程入口及
`../engineering/context-arithmetic-formal-acceptance-v2.json`、`../engineering/context-arithmetic-formal-start-v2.json`。
以下原正式失败与修复等待描述为历史过程，不覆盖本段。

状态以[当前执行入口](../engineering/context-prob-v1-baseline.md)与独立终态为准。
Stage1完成634更新；初末完整9226选择集均收集，终点set-F1 73.8041%、top1 82.0724%。
这不是RL或TEST成绩。连续两步、独立恢复、checkpoint interval1/64对照均通过。
正式GRPO来源绑定及运行包发布已完成，但容器19f100b20fd744d9在首个optimizer更新前
因binary log-prob一致性失败退出1/非OOM，完成0更新，634步曲线仍待。
终态详见`../engineering/context-formal-behavior-parity-failure-v1.json`；启动收据保留历史。

原序列在host来源绑定时漏传PYTHONPATH而失败；修复仅为新启动环境，复用同一冻结配置、
全部Stage1/选择集/工程验收产物，不改训练代码、奖励或预算。sequence v4及其comparison v6、
FSC v2、reference v3已因训练失败退出；原批次算术诊断与受影响修复验收完成前不重排。
后续两算法须使用同一修复版本、同Stage1起点和预算，不能只修GRPO而让对照运行旧数值路径。
原128条音频的单卡NPU重放及独立复算现已完成：7680概率精确相同，旧log路径最大差
4.0531158e-6；候选统一概率算术log差0，梯度最大差1.78814e-7。见
`../engineering/context-probability-arithmetic-npu-v1.json`。这只定位并验证小张量候选，
不代替主干反向、DP4更新和独立恢复；原冻结文件与阈值均不改。
原序列及其下游失败记录保留，不以历史状态冒充当前进度。

数值修复的真实DP4连续两步已独立审核通过，证据
`../engineering/context-arithmetic-grpo-prefix-v2.json`；四rank行为log差为0，
反向、更新、策略激活及执行/导出身份核验通过。新进程恢复容器813de6ba21b28ffa
已启动并精确等待，尚无重放比较终态。正式新版发布还须把相同数值协议贯通
运行、checkpoint和审核入口；不改任务、奖励、预算，不从工程两步checkpoint开始正式曲线。
原364冻结身份再次核验通过；本轮有限验收不提供RL提升或稳态吞吐证据。

一套共享工程，evaluation / instruction / task三用途，task只切换GRPO与RLOO-clipped。
PPO-critic仅设计；复杂Q、micro32吞吐调参及FSC不成为当前简单任务的发布前提。
任务/数据/损失/reward/batch/精度变化须先重新设计并登记新实验，不原位拼接曲线。
编排与收集只修复真实阻断点，复用成功产物；超时不是终态，不重建已有句柄。
普通远程观察至少600秒，准确docker wait继续持有原设备租约。

以下为历史记录及方法论证；其中旧阶段描述不能覆盖本节和当前执行入口。

## 历史核验记录（时间较早，不代表当前进度）

后续到期观察remote1794684604.146284确认原selection v2 running/nonOOM、
3136/9226；下一次普通观察>=1794685204.146284。原六队列本地仍存活。
当前尚无完整选择集指标或正式RL收益，不改变任务/算法配置。

最新更新：全量Stage1已完成634步并独立收集，exit0/nonOOM；正常步平均17.54秒，
约7.29 query/s。新概率头完整selection仍待，不能报告正式RL收益。
IPC过长、绕过Ascend入口、遗漏只读依赖三项启动缺陷已用独立版本修复；
真实固定镜像CPU导入/完整9226加载通过，新版编排45项CPU测试通过。
冻结364项、序列498项、比较500项身份复核通过，数值配置未变。
当前驱动548477/549655/549657/549658/550847/550848本地存活；旧518938及
533061因前序真实失败退出。最新远程确认selection v2 running/nonOOM，
observed=1794683960.6309571，next>=1794684560.6309571；观察脚本后续NameError
不代表模型失败。依据`docs/engineering/context-runtime-launch-repairs-v1.json`。
以下核验记录为历史；当前执行顺序为完整selection、既有有限NPU验收、
同起点GRPO完整曲线、RLOO-clipped完整曲线。不追加新的简单任务发布门槛。

2026-11-15（北京时间）本次本地锁校验：364 项通过，lock ID
`2ece49811d1e11eabc75f3e95e73ef532bda3fe9f16e9a399da7725ad0d95bfc`，
Stage2 released=false。文件身份锁定不等于完整 RL 运行验收通过。
最近到期远程观察为 1794681437.851721：Stage1 四 rank 完成 step618/634，
容器 04dd31b748d4 running/nonOOM；下一次普通观察不早于 1794682037.851721。
八个原训练及验收驱动保留；新增条件接续 driver518938 等待原选择集与checkpoint链，
不提前派发正式RL，也不重启原训练。

固定镜像无NPU CPU回归v3已独立收集：134 tests通过/36.51秒、exit0/nonOOM。
v2失败来自CPU测试AdamW自动探测NPU；夹具显式禁用foreach/fused后通过，生产
trainer原有相同设置，冻结代码未改。证据`docs/engineering/context-rl-training-audit-cpu-v3.json`；
该结果仅限CPU工具回归，不能据此开启正式RL或声明连续NPU训练已经验收。

当前全量作业是新概率头 Stage1：pairwise ranking + binary NLL，不是 RL。
GRPO/RLOO 有真实音频单步更新证据，尚无新概率头连续全量 RL 收益曲线。
历史 CE 分支的 9226 条 selection accuracy=82.3651%、macro-F1=79.5706%，
不能充当新概率头成绩、RL 收益或官方 TEST 成绩。

## 先固定任务及可证明的结论

输入为音频、完整候选定义及统一问题规约；浅 Transformer head 读上下文，
为 60 个候选输出独立 sigmoid 概率，供 harness 按冻结规则使用。
首轮只判断 exclusive_event：是否是 benchmark 标注的目标事件。
SLURP 单正例只支持该任务，不推出其余意图在语音中不存在，也不补造槽位负标签。
模型不生成文字思维链，不串接 ASR→文本模型，不将并行 Span 冒称串行推理。

学习人口 40513、选择人口 9226；TEST 不参与优化和选模。主指标为阈值 .5 下
query macro set-F1；同时报告 top1 accuracy/macro-F1、NLL/Brier、空集/多选率。
排序区分性和概率质量分别评估，奖励增加不自动意味着概率校准改善。

## 一套工程，按用途和算法切换配置

共享固定镜像、模型/processor、音频预处理、候选契约、LoRA/head 联合策略身份、
输入缓存、checkpoint、恢复、评测和日志。训练复用 verl actor/优势估计/surrogate，
rollout 复用 vLLM；概率动作桥接和生命周期编排是本项目适配，不能称原生 trainer
未经修改即可完成此任务。评测、Stage1、Stage2 是三个用途配置，不维护三套训练栈。

固定 Qwen3-ASR-0.6B，q/v LoRA r4/alpha8，BF16 执行和导出，FP32 master/Adam/loss。
175 卡4–7 DP4，global batch128 = 4×micro8×accum4；每阶段两轮634更新。
micro32 等吞吐尝试另立配置，先验容量及训推数值一致性，不原位改变当前曲线。

算法配置锁定如下：

| 配置 | 用途 | 估计器与边界 | 当前状态 |
| --- | --- | --- | --- |
| GRPO | 正式 RL 主线 | 同 query 的 G16 奖励组内中心化、标准差归一化，无 critic | 两步连续及独立恢复已收集；连续全量未发布 |
| RLOO-clipped | 同预算对照 | 其余 G−1 个奖励均值作 baseline，不作组内标准差归一化；复用 clipped surrogate | 单步已验；连续全量未发布 |
| PPO-critic | 预留研究配置 | 需独立 value 训练、保存恢复及验收 | 未实现，不作为首轮发布依赖 |

方法来源：GRPO https://arxiv.org/abs/2402.03300 ，RLOO
https://arxiv.org/abs/2402.14740 ，PPO https://arxiv.org/abs/1707.06347 。
这是方法的概率决策适配，不是原论文配方复现；RLOO-clipped 也不是未修改的 REINFORCE。

一次模型前向得到 p，采样 G16 个完整60位动作；不重复16次相同音频前向。
joint logπ(a|x)=Σk[ak log pk+(1−ak)log(1−pk)]，用完整动作的 likelihood ratio
进入 clipped surrogate；G16 不是16倍真实 query batch。
奖励为完整集合 F1，不用容易被59个负例支配的逐位准确率。

Stage2 总损失保持 L_RL + .01 KL(πθ||π_Stage1) + λrank Lrank + λprob LbinaryNLL。
KL 按候选求和、每 query 计一次；两辅助系数均从 .3 在前80%更新余弦衰减至 .1。
clip=.2、dual-clip=3、每行为批次一个更新 epoch。零信号组保留，不偷偷改变曝光分母。
Stage1 终点同时是固定 reference 和两个算法的共同初始化，之后不随 actor 移动。

## Step-wise：先调整任务奖励，不直接改标准算法

### 估计器差异与迁移适用范围

源码实查：`span_group_advantage.py`调用冻结verl的
`compute_grpo_outcome_advantage` / `compute_rloo_outcome_advantage`；
`span_sampled_hybrid.py`共用`compute_policy_loss_vanilla`；
`context_native_actor.py`复用`DataParallelPPOActor`的更新/优化器循环。
类名或配置里的PPO不代表当前启用了critic-PPO；当前正式主线登记为GRPO。
概率动作桥接由本项目实现，不是原生文本生成trainer直接替换输入即可运行。

对同一输入的G个完整动作，GRPO优势为`(R_i-mean(R))/(std(R)+epsilon)`；
RLOO为`R_i-mean(R_j, j!=i)=G/(G-1)*(R_i-mean(R))`。
因此固定G16时，本对照主要检验组标准差归一化及优势尺度带来的影响，
不能将它描述为两种完全不同的推理或节点信用分配机制。辅助loss与KL系数
保持相同，优势尺度变化会改变它们相对RL梯度的影响，报告中须说明。

参考原文：DeepSeekMath §4.1.2/§4.1.3
https://arxiv.org/html/2402.03300v3 ，RLOO
https://arxiv.org/html/2402.14740v2 ，PPO
https://arxiv.org/abs/1707.06347 ，verl官方配置说明
https://verl.readthedocs.io/en/latest/algo/grpo.html （本次访问403，未作为已读证据）。
本次原论文及本地源码复核日期2026-11-15；
网上文档仅用于方法依据，运行数值实现仍由本地源码hash固定，不随latest更新。

### 复杂任务的奖励和信用分配

输入 Q1/Span1、Q2/Span2、Final/Span3 的并行一次填充没有真实时间状态转移。
不沿 Q 编号使用 temporal GAE，不将后面 Span 当作前面采样动作造成的环境状态。
单层并行 head 的 Final 也不会自动消费同层刚输出的其他概率。

首轮复杂任务比较限定为：

1. Direct：直接最终判决。
2. Q-input：增加统一语义规约问题，仍只用最终任务 reward。
3. Q-credit：与 Q-input 使用相同模板、动作、过程辅助监督及预算，仅增加有可靠标签的过程 reward。

Direct 对 Q-input 测量整个问题结构方案；Q-input 对 Q-credit 隔离过程奖励贡献。
过程标签缺失时 mask，不能填 false；Final 标签的确定性投影明确称辅助属性监督。
复杂任务的过程权重需在设计和数据支持核验后另行冻结，不改变当前简单任务配置。

先用 R=R_final+λR_process 接入现有 GRPO/RLOO，不新增算法名称。过程项先在
冻结任务族内归一化，避免增加问题数量就增加 reward；按缺标覆盖分别报告。
GRPO 的组标准化可能放大很小的过程差异，λ 小不等于过程梯度占比小。
具体而言，若组内Final奖励恒定而过程奖励有变化，在epsilon可忽略时，
`normalize(R_final+λ*R_process)=normalize(R_process)`（λ>0），
所以不能用“把λ设得很小”保证Final信号占优。先报告这些组的比例和实际
梯度/优势尺度；是否采用分项优势及统一固定缩放，应另立后续算法变体，
不能静默改写当前GRPO。全组无差异时RL优势为零，保留该组，辅助监督仍可学习。
必须报告 Final 恒定组比例、组 reward 标准差、有效优势组比例、clip fraction、
KL、概率饱和、过程与最终指标及 NPU 小时收益；不预先声称已有全部诊断埋点。

只有这些证据指向联合信用分配确实阻碍最终收益，才新建节点条件信用实验。
可在奖励可计算时用离散动作反事实评价节点贡献，但替代动作必须符合合法动作域；
不得把逐节点 ratio/clip 或独立归一化当作原 GRPO 的无影响实现细节。
过程准确率提高而 Final 不提高不构成决策能力收益；因果多步推理还需单独干预证据。

DeepSeekMath的过程奖励传播针对实际生成序列及后续步骤；本项目固定Q并行
读出没有这种动作条件化的后续生成，不能直接照搬其token reward-to-go。
若后续引入PPO，本任务首先是单步上下文决策，其critic可预测每个query的
期望总回报；不能把候选索引当时间轴。当前单层head的Final读上下文隐藏态，
不直接读取同层刚输出的其他prob。若任务确实要求显式消费前置prob，需要
预先定义harness聚合或新的内部结构，然后再单独冻结工程与实验。

集合F1对应决策效用，不能证明概率已校准；排序项也只约束相对次序。
因此保持binary NLL辅助并单独报告NLL/Brier。当前RL结论限定为固定混合
目标下相对Stage2 step0的变化，不将全部变化归因于纯RL项；不额外增加
SFT对照来阻断本轮实验。

## 发布的有限剩余工作与顺序

1. 原 Stage1 已正常收尾；完成初末策略的完整选择集收集，不重启训练。
2. 已收集设备共驻、同 actor 连续两步、策略刷新和新进程恢复重放证据；继续
   原队列中的checkpoint间隔验收，检查interval1/64逐步策略及末步master/Adam。
   不重做已通过的连续/恢复探针，不把早期Resource_Busy归因于已证实的镜像损坏。
3. 正式634步runtime、保存和11点评测编排已经接入当前v3序列；待既定真实
   输入齐备后完成来源绑定并封存Stage2发布。CPU编排通过不能替代实际NPU曲线，
   也不能只切换dispatch开关跳过已有准入。此后不扩展其他前置功能。
4. 正式 GRPO 634步；同 Stage1 起点/reference/数据顺序/曝光/G16/优化器/辅助项
   运行 RLOO-clipped。完整9226 selection 在 step0、64…576、634共11点评测，
   展示全曲线和退化，不仅挑最好结果；不新增 RL-vs-SFT 比赛。
5. 简单任务闭环后再发布有监督支持的复杂问题任务，并做 FSC 冻结权重配置迁移。
   PPO-critic、复杂问句和迁移均不成为简单 RL 发布的前置条件。

本次将 checkpoint 生命周期调整封存为独立候选：每步继续导出和激活行为策略，
完整 master/Adam checkpoint 仅在预定间隔及运行段末保存。10项CPU测试通过，
包括实际变换后的保存代码块及恢复后模型/Adam一致性；FSDP、通信和导出器使用替身，
不能替代NPU验收。原两步守卫保留，未改排队源码或派发正式RL。
证据：`docs/engineering/context-checkpoint-runtime-cpu-v1.json`。

已冻结源码/数据/损失/精度/batch 改动必须产生新实验身份；候选修复不能写回运行包。
当前无需用户重复确认算法或授权。完整连续 RL 尚未验收，不能承诺已经有稳定生产基线。

正式起点绑定候选：`tools/jev/context_stage2_source.py`只接受完整instruction step634
及对应已收集selection，不接受两步探针或中间最优点。两算法复用相同终点与固定reference，
任务阶段重置优化器；训练和评测policy ID可以不同，但实际张量、processor和adapter必须相符。
绑定结果本身不授予Stage2准入。41项CPU测试和真实既有prefix张量认证证据见
`docs/engineering/context-stage2-source-cpu-v1.json`；完整终点已收集，selection仍待，未执行实际全链绑定。

该绑定已接`tools/jev/prepare_context_stage2_source.py`准备入口：接受同样的`--spec`，
以`--output`指定新的本地目录，打包原Stage1策略及其processor，并重新派生对应原
policy ID的推理adapter；两算法共用该包，reference不变，任务阶段重新初始化Adam。
没有remote调用或发布授权。22项相关CPU测试通过；实际Stage2共同起点包仍待完整selection。

正式运行入口现有独立候选：`prepare_context_formal_runtime.py`从精确checkpoint
候选只替换准入检查，不修改原数值代码。`context_formal_acceptance.py`从独立收集
重验连续/恢复/间隔checkpoint证据，供后续发布器消费；尚未以真实NPU终态执行。
正式package发布器与launcher候选现已接入上述来源/验收接口，共48项本地测试通过；
固定镜像79项CPU回归通过。完整Stage1已有，完整selection及连续NPU证据仍待，因此未发布正式包或
派发RL。收尾流程见`docs/engineering/context-formal-training-runbook-v1.md`，后续完成
真实输入发布、训练收集/audit和11点评测接续，不新增任务/算法前置条件。

2026-11-15 收口补充：完整实验编排固定镜像CPU v2已独立收集118 tests通过/36.59秒；
v1遗漏动态导入的测试夹具导致16项失败，补齐测试包后通过，原失败证据保留。
`run_context_baseline_sequence.py`已把共同起点发布、GRPO完整曲线和RLOO完整曲线
接在原Stage1选择链与checkpoint验收链之后。7队列+14实验编排tests通过；实际训练
和SSH在测试中使用替身。467文件pins已锁，driver518938目前仅等待本地前序锁。
前序失败或源码漂移即停，不自动重跑。正式Stage2仍未发布，不能称已有RL收益。

两算法完整曲线后的只读比较已接续：`compare_context_rl_curves.py`先重审实际训练
及每点评测，确认共同起点逐条概率、source资产、数据日程、精度及优化器一致，才
输出逐点差值、最终/选中点增益、退化与成本。NPU成本按容器wall×4物理卡计，
service共卡不重复相加，当前比较不包含评测成本。单次运行描述不作显著性或纯RL
因果结论。26新+12既有38项CPU测试通过，实际模型/audit在集成夹具中替代。
driver533061（v2）等待原双算法序列成功后执行比较，469pins已复核；不追加训练门槛。
原528969仅等待时退出归档，修正正式产物evaluation/selection路径后38项CPU测试通过，
证据context-rl-curve-comparison-cpu-v2.json；训练和原双算法序列未改。

## 本次执行收口规则

共享一套工程，由评测、instruction和task配置选择用途，算法只切换已登记估计器。
稳定版身份由源码、镜像、模型、processor、数据、日程、损失及评测配置共同决定。
本轮核验364项身份均通过，但stage2_released=false，不能称完整RL稳定版已经发布。

仅处理阻断当前训练/评测闭环的缺陷；修复在独立候选中验证并保存旧证据，
不修改运行包、不把修复前后混为一条曲线。吞吐调优、PPO-critic、FSC和复杂Q
均不再追加为简单任务发布的前置条件。现有条件序列通过原验收后接续，不重复派发。

简单任务的最终判据是相同RL起点下的完整11点选择曲线，以及NLL/Brier和成本。
复杂任务先确定可迁移Q、合法动作、可用过程标签、缺标mask和Final评分，
随后冻结Direct/Q-input/Q-credit实验，再实施工程变更和训练。
过程reward按任务族及有效标注归一化，并分层报告覆盖率；过程标签不足时
只允许Q-input，不以推测标签启动Q-credit。增加问题长度同时计入token及NPU成本。
只有联合优势稀疏、过程主导或联合ratio裁剪的实测诊断表明信用分配受阻，
才新增节点信用算法版本。该版本与原GRPO/RLOO分别命名和比较。
