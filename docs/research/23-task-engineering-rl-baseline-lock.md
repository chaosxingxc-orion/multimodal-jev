# 任务先行、统一工程与RL配置基线

> 当前概率头实验以 [context-prob-v1权威入口](../engineering/context-prob-v1-baseline.md) 及其冻结清单为准；下文含历史Final60/CE与任务候选，不自动覆盖当前执行配置。

## 当前执行决议（替代下方历史运行状态）

执行顺序为任务定义与比较设计 → 公共工程和配置冻结 → 受影响链路有限验收 → 全量训练与评测。
已准入的简单任务继续收尾，复杂任务先完成自己的设计和验收，不反向阻断简单基线。
只维护一套 verl/vLLM 工程，通过配置提供评测、Stage1、Stage2 三种用途。

当前实际运行是 arithmetic-v2 的 GRPO，无 critic；RLOO-clipped 已封包排队，PPO-critic
仅设计、尚不可运行。源码复用 verl 优势估计与 clipped surrogate，typed Bernoulli 动作
和常驻编排是本项目适配，不能称原生 token RL 配方的无修改复现。当前算法、损失、
精度、batch 与运行包不原位调整。基线清单本轮重新核验 364 项身份全部通过；清单中的
旧 stage2_released=false 是历史状态，正式发布以 arithmetic-v2 验收与运行收据为准。

简单任务锁定原始音频和完整 60 候选定义，一次读出 60 个独立概率，harness 阈值 0.5；
全量 TRAIN 40513、selection 9226，TEST 不参与优化和选模。Qwen3-ASR-0.6B、共享浅层
Transformer head、q/v LoRA r4/alpha8、BF16 执行、FP32 master/Adam/loss；DP4、
global128=micro8×accum4×4。Stage2 固定 G16、634 更新、11 点完整 selection。
Stage1 已完成；GRPO 最近归档进度统一见权威入口，不在本设计文档复制易失效的步数，
运行中的训练进度不能作为 RL 效果结论。

复杂任务首轮以 `configs/experiments/context-native-attribute-comparison-design-v1.json`
为准：Direct60、Q-input124、Q-credit124（18 scenario + 46 action + 60 Final）。
Q-credit 使用 Final_F1 + 0.25×(scenario_F1+action_F1)/2，两 Q 组只差过程 reward；
三组均只保留 Final 排序及概率辅助损失，不加入过程辅助 loss。下方历史条目中的
过程 CE、权重 0.2 及旧主指标不覆盖该比较。过程标签是原生 Final 的属性投影，不是
独立语义推理 gold。全量输入审核与真实 Qwen/NPU 训推、更新、新进程恢复验收仍待闭合。

首轮保留标准 GRPO/RLOO 优势估计和联合动作裁剪，先回答过程信用是否提升 Final。
GRPO 在 Final 恒定组内会将非零过程奖励重新标准化，小 lambda 不保证小过程梯度；
RLOO-clipped 无组标准差归一化，是检查这一问题的匹配对照。两者的恒奖励组优势均为零。
只有完整曲线及信用诊断支持后，才单独登记分项优势或节点反事实信用变体。
并行 Q 不是时间步，不沿 Q 编号套 GAE；前置概率没有回灌 Final，不能声称已实现显式概率链。

稳定工程的收口标准是固定身份及有限真实验收；学术成功仍需完整 Final 曲线、概率质量、
TEST 与配置迁移证据。当前优先完成 GRPO → RLOO-clipped → FSC/clean TEST/MINDS 队列。
不将 PPO、新复杂任务、更多诊断或 micro32 优化追加为当前简单基线的前置条件。

## 封版后的变更与算法准入

任务协议、公共运行时、实验参数分别版本化，并在每次运行清单中共同绑定。仅记录
进度和分析已闭合产物不产生新运行时；修改输入模板、类别映射、采样、损失、精度、
数据人口、batch或优化器都必须形成新实验身份，不能拼接到原曲线上。
候选rc1遇到阻断缺陷时保留原证据，生成rc2并执行受影响验收；已通过范围不因普通
诊断重复扩展。当前不再新增训练器，也不要求PPO或复杂语义任务完成才能发布简单基线。

算法优先级依据当前任务：GRPO无需critic，适合可核验分类奖励；RLOO-clipped复用同组
采样与同一surrogate，主要检查组内标准差归一化是否影响稳定性；PPO-critic增加value
网络及其拟合和恢复负担，暂不接入。GRPO/RLOO在全错或全对采样组都没有组内相对奖励
信号，因此更换这两者不能自动解决奖励稀疏；先完成Stage1，再依据正式曲线判断。
过程奖励可能让Final恒定的组获得优势差异，但也可能只提高辅助判决；是否升级算法，
以固定选择集Final收益、KL与单位NPU时收益为依据，不以过程reward上涨作结论。

## 算法原始来源及适配范围（2026-11-13 UTC）

- GRPO：DeepSeekMath，arXiv:2402.03300v3，4.1.1–4.1.3节。组内中心化/标准差
  归一化与不使用critic来自原方法。本项目固定typed动作、精确类别KL、辅助CE、
  dual-clip=3及每行为批次单次更新，属于明确登记的适配，不是论文训练配方复现。
- RLOO：Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from
  Human Feedback in LLMs，arXiv:2402.14740，2.3节。原始估计器使用其他G−1个
  样本奖励均值作baseline；本项目复用clipped surrogate，名称保留RLOO-clipped。
- PPO：Proximal Policy Optimization Algorithms，arXiv:1707.06347v2，3、5节。
  PPO-critic作为预留配置；必须实现value训练/恢复并单独验收后才能称可运行。

DeepSeekMath的过程GRPO在生成序列的推理步骤末尾计分，并对后续步骤的标准化奖励
求和。本项目的Q/Span是输入预设、一次联合填充，没有已采样答案驱动下一环境状态的
事实，不能直接移植该时间信用定义。首轮复杂任务先固定联合动作与可审核过程奖励，
沿用现有优势估计；节点条件信用需另立变体与干预实验，不改当前简单Final60发布。
以上是方法来源与本项目设计边界，原论文结果不构成本项目语音任务有效性证据。

## 执行约束与研究问题的收口

锁定一套公共工程及多个算法配置；不为每个算法维护一个训练栈。简单任务按既定定义
继续Stage1，正式Stage2须完成工程验收后发布。当前实际全量作业是CE指令适配，
19:37 UTC到期观察原容器仍运行，四rank已完成310/634更新、正在311；GRPO仅有工程更新
证据，尚无正式全量RL收益曲线。运行快照的progress.step为零基，不能再减1计完成量。

统一候选rc1已经封存：1355份Python源码、2份权威配置，归档SHA256
`6194e16d3dce93c3aed982bd486ed9472adff7e924e218d323786fdbd1fabe5f`。
同一固定镜像CPU回归275项全部通过、无跳过；18:49 UTC再次核对全部1357份文件与归档
均未改变。收据见`../engineering/slurp-simple-rc1-cpu-acceptance.json`。
上述rc1归档保留；当前工作树已进入rc2，不能继续引用rc1证明当前源码通过。
rc2补齐连续checkpoint恢复后再更新的入口与正式准入检查，141项本地回归通过；
1360份源码/配置已封存，19:39 UTC到期确认固定镜像CPU exit0、304项通过、无跳过；
独立核对全部源码/配置/归档/runner一致，当前工作树仍匹配rc2。真实NPU验证仍待执行。
记录见`../engineering/slurp-simple-rc2-cpu-acceptance.json`。
这完成了rc1同版本CPU验收，尚不构成真实常驻NPU生命周期验收或正式Stage2发布；
以下历史条目中的待验状态以本段及当前状态页为准，不通过修改冻结配置登记进度。

公共工程固定Qwen3-ASR-0.6B、BF16语言q/v LoRA r4/alpha8、FP32 master/Adam、
verl FSDP DP4与vLLM TP1；简单任务全量40513/9226来源隔离，global batch128。
复用的是固定verl的actor更新循环、优势估计和surrogate，typed-action桥接及常驻
编排为本项目实现；不能把组件通过称为完整原生verl trainer的无修改验收。

**发布剩余条件限定为三项：**

1. 最终Stage1四分片及导出审核、完整9226选择集评测和训推一致性。
2. 同一actor/常驻vLLM真实连续更新、独立审核、新进程精确恢复及再次更新。
3. 将通过验收的源码hash、配置、模型/数据/日程身份和启动/恢复入口组成不可变发布。

恢复范围澄清：现有continuous-restore探针只验第三步状态及128条概率，不执行更新；
不能单凭它声称同一连续checkpoint新进程恢复后再次更新已验收。该能力的真实NPU
证据仍待补齐，详见simple-stage2-runbook的交接说明。

现有Stage1不因配置研究而重启；新算法、复杂Q及吞吐扩展不进入上述发布条件。
通过后先运行GRPO全量曲线，再在相同起点/数据曝光/预算上运行RLOO-clipped。
PPO-critic仅保留设计角色，未实现前不标成可运行基线。影响数值的修订形成新版本
及新曲线；验收后的默认路径不因常规实验继续改工程。

step-wise当前指预先给定语义问题的一次联合判决。首轮不修改GRPO/RLOO优势估计器：
在同一完整动作上使用最终奖励与有监督的过程奖励，缺标mask，过程项按冻结任务族
归一化且限定总权重。过程问题若只是Final标签的确定性投影，只能称辅助监督，
不能当作独立多步推理证据。节点信用/节点裁剪另注册变体，不沿Q编号使用时序GAE。

复杂语义轨当前设计入口为`reward-contract-design-v3.json`，仍未准入运行，不能覆盖
简单Final60配置。Q-input与Q-credit保持相同过程CE、缺标mask与衰减，只改变过程reward，
才能隔离奖励的贡献。GRPO组内标准化可能把微小过程奖励差异放大，原始过程预算0.2
不等于梯度占比上限；须报告Final恒定组比例、reward标准差和优势/梯度尺度。
仅在这些诊断及Final收益支持时，再注册条件节点信用等新变体，不静默修改标准算法。

先比较Direct、Q-input、Q-credit；Q-input与Q-credit保持相同模板、动作空间和预算。
主判据仍为Final accuracy/macro F1，辅以过程准确率、有效奖励组比例、KL与单位NPU时
收益。过程准确率改善而Final无收益时，不宣称决策链有效；若要声明中间判决影响Final，
需要表示/决策干预证据，联合准确率上涨本身不足以证明。

本轮补充末批与终态分段审核：29项本地CPU测试通过，覆盖epoch交界与634终态、
128槽位/65真实请求、错误分母/Adam/终态游标拒绝。修正的是测试替身请求数；
模型/音频/前缀来源/selection使用替身，非完整634步执行或NPU验收，不改生产快照。

## 已闭合GRPO两步的信号诊断（2026-11-13 19:05 UTC）

按独立审核中的原始文件hash重新读取两次工程更新，完整128样本×16采样，关联冻结
learning标签及固定Final60顺序。首步复算与原独立审核计数一致，第二步新增同口径统计。

| 更新 | 全错组 | 混合奖励组 | 全对组 | 正确采样/2048 |
| --- | --- | --- | --- | --- |
| 1 | 109 | 19（14.84%） | 0 | 30 |
| 2 | 106 | 22（17.19%） | 0 | 29 |

两批样本及行为策略不同，起点仅两步CE工程前缀，不是最终Stage1；混合组增加不等于
模型提升。全错组的组内RL优势为零，CE仍提供监督；不能过滤掉这些组改变人口分母。
四rank日志复算RL损失约−1.16e−10/1.98e−9，中心化优势在ratio≈1处使标量近零，
不能据此断言策略梯度为零。CE未乘衰减系数的加权均值7.6925/6.9046；已乘0.01的
KL贡献约−6.68e−11/0.001326，首步微小负值是数值残差，不据此修改目标。
总梯度范数24.85/13.10不提供RL/CE各自贡献；当前证据不证明RL单独有效或无效。

固定配置不因这两批诊断改变。正式RL须以最终Stage1初始化，记录完整曲线上的有效组
比例、奖励/CE/KL及Final效果。证据`../engineering/slurp-simple-grpo-fixture-signal-v1.json`；
只读闭合产物分析，没有新音频推理、NPU训练或rc1修改。

## 本次收敛决策（2026-11-13）

**配置规约已锁定，正式Stage2工程发布尚未完成。** 当前全量运行是Stage1 CE，
实际已执行的RL工程探针是GRPO+辅助CE+固定参考KL。以下是本轮唯一执行范围，
下方时间记录只作证据历史；不再以新增算法、数据任务或吞吐候选扩大发布前置范围。

| 顺序 | 固定交付 | 完成标准与当前缺口 |
| --- | --- | --- |
| 1 任务冻结 | clean SLURP全40513 learning；9226来源隔离selection；音频+固定60类定义→单Final | 已冻结；缺陷排除清单、标签映射、失败分母和指标随release固定 |
| 2 公共工程验收 | 同镜像/模型/processor/LoRA/数值profile；verl DP4训练、vLLM TP1推理 | 前缀、一次GRPO更新、新策略概率、驻留切换与新进程恢复已验；两次更新检查点和新策略概率已通过；常驻连续循环待收尾 |
| 3 实验发布 | 固定源码清单与hash、配置、全量日程、启动入口、恢复入口、证据索引 | 每次运行消费不可变快照；工作树本身不算已发布基线；连续常驻全量RL trainer仍需完成 |
| 4 正式Stage2 | 最终Stage1权重作起点及固定KL参考；GRPO先行 | 待Stage1终态/selection和上述工程验收；不能将两步CE前缀fixture用作正式Stage1 |

一套基座保留三个算法角色：GRPO主配置、RLOO-clipped比较配置、PPO-critic预留。
只有前两项已有固定镜像CPU实现；PPO-critic尚未实现，不为凑齐算法数量复制trainer。
RLOO只替换完整采样组的优势估计，复用音频输入、actor、采样、CE/KL、导出和评分。
算法对比固定起始checkpoint、数据曝光、G、更新预算及selection时点，另报NPU时与吞吐；
相同预算不意味着两个优势估计器具有相同梯度尺度，分别记录优势/梯度范数。

效果判据使用预登记selection时点的Final accuracy、固定60类macro F1、逐类召回与失败率，
同时展示整个曲线及末点，不能只报最高点、训练Reward或要求每步单调上涨。
持续提升目前是待验证假设；单次更新成功及混合loss下降都不能证明RL产生收益。
训练日志分别报告RL/CE/KL、非恒奖励组比例、行为与当前策略差异和梯度有限性。
这些是解释混合训练的诊断，不新增用户未要求的RL与SFT胜负实验。

复杂任务单独按“Q语义/可获得监督/输入输出/指标→工程容量→训练”推进。
Q由部署时已知任务定义构造，缺标mask；一次同时判决不是前一采样答案被下一Q消费。
先用Direct、Q-input、Q-credit验证结构和过程信用收益，再决定是否注册算法变体；
不提前修改GRPO/RLOO估计器，不沿Span编号使用时序GAE。

工程修复只允许解决当前准入失败，并形成新快照与定向回归；不得原位修改运行任务。
吞吐优化独立登记，已有micro2短测收益不自动升级正式曲线。
HF独立对齐用于准入/验收；正常resident行为采样可不重复跑HF，但每次更新仍核对
actor/behavior概率与版本。真实vLLM切换与无HF输入行为已独立验收；常驻两批输入
已验收；同一actor连续更新仍待验收，不据此声称全量RL效率或所有更新的训推一致性已经达标。

## 配置权威与算法边界（2026-11-13）

简单主线的运行参数只从 `slurp-simple-final-v1.json` 和
`slurp-simple-sampled-rl-v1.json` 读取。`rl-baseline-design-v2.json` 是复杂任务研究候选，
其中两次optimizer epoch、dual-clip为空等配置不能覆盖简单主线的一次更新、dual-clip=3。
“配置锁定”指确定这份实验规约；正式发布还必须附源码快照hash、验收证据及启动/恢复入口。
不将含有工作中修改的目录整体称为稳定发布。

| 算法配置 | 优势与损失 | 当前适用边界 |
| --- | --- | --- |
| `grpo_v1`，主线 | 同输入G16完整采样组，调用固定verl的GRPO，组内中心化及标准差归一化；clipped surrogate + CE + exact KL | CPU与两次真实NPU工程更新已验；持续常驻训练待验 |
| `rloo_v1`，比较 | 每个样本以同组其他G−1个reward均值作baseline；保留相同clipped surrogate/CE/KL | CPU已验；称RLOO-clipped，不能声称原始无裁剪RLOO的严格复现 |
| `ppo_critic_v1`，预留 | 学习V(audio, task)，一步return−V及value loss | 未实现/未验收；不阻塞主线，不按Q编号构造GAE |

简单任务的loss固定为真实请求均值：
`L = sum_x w_x * (mean_g L_policy(x,g) + lambda(t)*CE(x) + 0.01*KL(pi||pi_ref)(x)) / N_real`。
Final正确奖励1、错误0；过程奖励0。类别权重由全量learning人口冻结，均值1，
不在microbatch内重新归一化；padding在所有项上零权重且不计N_real。
CE系数在前80%更新按cosine从1衰减到0.1；参考策略为最终Stage1，整个Stage2固定。
全错/全对组保留，它们的组内RL优势为0，CE仍可提供学习信号。中心化后RL loss标量
接近0不能解释为梯度0，必须记录非恒reward组比例和策略梯度诊断。
每个行为批次仅更新一次且更新前actor与behavior一致，因此该次梯度处ratio≈1；
clip不限制第一次optimizer移动幅度。不得仅以存在clip参数宣称已控制更新幅度，
须报告更新后KL、梯度范数和selection效果；调参作为新配置，不能原位改曲线。

复杂任务先冻结可部署Q和独立监督，再注册以下受控实验：

1. **Direct**：音频+任务定义→Final。
2. **Q-input**：固定语义问句及其Span加入输入输出，只奖励Final。
3. **Q-credit**：与Q-input完全相同的输入、动作坐标和预算，加有监督的过程信用。

Q不得从当前样本gold反推；例如“是否出现定义中的槽位类型”必须先有该定义及其可复用
判决规则。缺标mask而非补负。过程奖励按冻结任务族权重/容量归约，不能随问句数增加
无界放大，也不能奖励多个等价问题重复计分。先比较Q-input与Q-credit，避免把输入变化
误称过程奖励收益；Direct对比另报问句数、KL/联合ratio几何与计算成本差异。

首版复杂任务仍用完整动作的GRPO/RLOO，先不修改优势估计器。若存在可靠逐节点监督，
再把节点优势与节点surrogate作为独立变体；逐节点裁剪不等于联合动作裁剪，不能悄悄
替换标准基线。一次前向同时填Span没有“前一答案改变下一状态”的转移，过程信用
不是时间回报。若要证明中间判决被Final使用，还需对中间表示/决策作受控干预或移除，
不能仅凭过程准确率和Final同时上涨声称存在决策链因果收益。

本轮恢复启动器补齐7项实际入口测试：已存在句柄/目录、卡占用、失败来源、缺独立验收、
hash不一致均在远程写入前拒绝；成功路径核原审核路径、源码hash、只读挂载与DP4设备。
SSH/Docker为替身，属于本地入口验证，不是NPU恢复成功。

## 正式主机派发与收集入口（2026-11-13 18:22 UTC）

固定镜像主机入口CPU `84157ce4…` 于18:31 UTC按600秒间隔观察exit0并独立收集：60测试通过，1355份源码hash与派发归档完全一致。该快照早于service/trainer恢复游标一致性补充和本轮末批测试；当前工作树主机入口29项、分段审核29项本地CPU测试通过。SSH/Docker、外部模型/音频使用替身，不是实际NPU验收。正式Stage2仍未放行。

恢复集成CPU `7aae66ca…` 于18:23 UTC到期exit0并独立收集：177测试通过，1351源码hash与归档一致；真实张量及合成全9226评测，外部模型/音频替身，非NPU恢复。该快照早于主机launcher/collector。

共用两阶段launcher增加--resume-audit-tag与唯一--run-id：保留原配置/参考/生命周期证据，
新目录/服务/训练快照，禁止覆盖旧任务。派发前实时核源容器exited，并在远端逐hash复查
已收集审核及其依赖；不完整逐步策略链/到期评测、损坏依赖、变化实验均在写入前拒绝。
恢复模块认证在torchrun前运行，模型加载后仍走同一actor概率准入。

新增launch_simple_stage2_audit.py独立CPU审核派发，显式prefix游标或完整终态634；
collect_simple_stage2_audit.py复查远端原始证据，完整保留报告字节与收据。
observe_due自动发现stage2-*-launch.json，保持每句柄至少600秒，不自动续派/重启。
具体启动/终态/中断恢复命令见docs/engineering/simple-stage2-runbook.md。

28项本地主机入口测试通过，实际脚本分支执行，SSH/Docker使用替身，不是真实NPU派发。
固定镜像CPU `84157ce4de36d59f65f77b0a7669e660778896c1ddebb2de330265ced181c529` 已启动，按launch600秒；上一resume CPU仍待自己的deadline。
远端tar约39–72秒mtime时差，仅提示；身份继续用内容hash，不因mtime重启。
派发后另补service/trainer恢复审核与游标一致性检查，启动器13本地测试通过，
含新增拒绝路径；该修改晚于host-launch CPU快照，不混记固定镜像验收。
正式开关false，尚待固定镜像结果、真实连续NPU/恢复、最终Stage1完整验收。
学术基线、RL持续收益与跨数据集泛化仍未完成，不把主机入口通过当目标完成。

## 正式恢复加载与分段审核（2026-11-13 18:14 UTC）

中断前缀CPU `081a1dc0…` 已独立收集：131测试通过，1349源码hash与归档一致。
该快照早于本轮加载/评测恢复/策略链/分段审核，不能扩展验收范围。

新增simple_stage2_resume把正式release、最终Stage1固定参考、已退出前缀来源串接。
共用actor加载器显式恢复FP32 master与Adam，保存stage2-resumed.pt；不清空RL优化器，
不把新策略变成KL参考。continuous共用循环从恢复游标开始，首次新行为仍执行actor
概率准入。ResidentSimpleSelection复算全部既有原始评测，再从下一预定时点继续；
不重跑第0点、不覆盖历史文件。采样随机种子已有update/slot/policy身份绑定。

独立审核支持分段：继承认证前缀的逐步检查及完整policy序列，只审核新段更新；精确
比较恢复初态master/Adam，历史评测沿原文件/四rank引用复算，完整终态仍要求634及
全部11评测点。新恢复源必须具有逐步policy链；旧未包含该字段的CPU替身报告不准入。

本地47恢复/状态/评测测试、94入口/阶段/循环/审核回归、110扩展恢复回归与4新增分段
测试通过，计数有重叠。含实际张量下一次Adam更新与不中断运行精确一致；完整9226
合成概率原始重算、外部模型/音频/来源认证替身，不是NPU验收。NPU路径仅语法检查。
固定镜像CPU `7aae66cadeda82936fb40e51c3e63291d84b398752e934ef7961027f902f568d` 已启动，按launch至少600秒观察。
尚缺正式恢复主机启动器/审核派发接入、真实NPU生命周期与最终Stage1验收；formal
false，已跑快照不变。学术基线复现、复杂语义任务与跨数据集收益仍未完成。

## 中断前缀审核与恢复来源（2026-11-13 18:05 UTC）

正式启动器CPU `1d2fbf66…` 已独立收集：47测试通过、1349源码hash与归档一致。
范围是发布认证、真实启动器分支（SSH/Docker替身）及阶段切换，早于本轮恢复修改。

发现旧恢复准入要求源任务exit0，无法处理训练中断。本轮增加显式stopped_prefix审核：
必须固定容器已经exited及确切整数退出码，可缺少因崩溃未写出的terminal.json；绝不
从缺文件或超时推断任务已停。完整终态审核仍要求634步成功，不被该模式放宽。
每个正式checkpoint额外持久化四rank BF16 hook证据；审核指定点前每次采样、状态、
导出与概率，并重算所有到期selection。未来失败后缀保留但不认证；缺到期评测则拒绝
该恢复点，需要明确选择更早的完整点。恢复来源验证保持原算法、参考gate、奖励配置、
评测日程和精确master/Adam游标，工程fixture不能冒充正式checkpoint。

本地78审核/重建/旧流程回归、4正式前缀集成、49恢复相关测试通过，计数有重叠。
实际张量、完整G16采样和概率检查执行；外部模型/音频/来源认证在对应测试中使用替身。
固定镜像CPU `081a1dc0d10ffa0ed3f583375afd170f6221c2781869ff8e632039e7cc4e58f5` 已启动，依launch至少600秒观察。
目前仅补齐审核与来源认证；正式恢复actor/评测游标/循环/启动器尚未接通，也没有实际
NPU恢复或完整634审核结果。formal dispatch仍false；原运行快照不修改。
Stage1最新18:03 UTC四rank正在第236更新，235/634完成。

## 本轮收口与已验收范围（2026-11-13 17:55 UTC）

当前运行Stage1在17:53 UTC观察已完成229/634、准备第230批；四rank一致，原快照继续。
常驻选择集CPU `ee061c1a…` 已独立收集：60测试通过，1344份源码与归档hash一致。
正式终态审核CPU `3a966fc0…` 已独立收集：82测试通过，1346份源码一致。
两者均有外部模型/音频替身，不是实际NPU选择集或完整634步审核验收。

正式两阶段启动器 `launch_simple_resident_train.py --formal`、发布认证及训练入口已实现；
16项发布测试和7项启动器测试通过（SSH/Docker替身）。阶段切换兼容原Stage1配置省略
microbatch字段时已有的默认值1，不改变运行配置。固定镜像CPU `1d2fbf66…` 待按自身
600秒间隔收集，不提前记为通过。正式恢复接入仍待完成，training_dispatch_allowed=false。

收口仅剩：实际常驻连续NPU更新与新进程恢复验收、最终Stage1及完整selection验收、
正式恢复入口和发布快照。PPO、复杂Q、新数据集及吞吐候选不增加到简单模式发布条件。
工程锁定以源码归档hash与配置/模型/数据/镜像身份为准，不以工作树或历史测试总数代替。
已运行版本不原位修复；任何影响数值的修订另发版本，定向验证后形成新实验曲线。

## 正式Stage2独立审核准备（2026-11-13 17:44 UTC）

阶段切换CPU `2abd7c8b…` 到期exit0，56测试通过，1340源码hash与派发归档一致；
证据stage2-initialization-cpu-verified.json。该快照早于常驻评测和本轮终态审核，不能
扩展其验收范围。Stage1最近17:41观察四rank在第221更新（前220完成），继续原快照。

新增simple_stage2_audit核固定镜像成功进程、最终Stage1依赖、四rank初始精确FP32
master与空Adam、完整BF16执行hook，并复核11个评测点原始计分与各rank引用。
原verify_simple_resident_train增加显式正式分支，复用完整采样/参数重建/概率审核，
范围1..634、末批真实请求分母来自schedule，stage=simple_stage2_rl与工程fixture隔离。
模型加载路径新增stage2-initial.pt和执行hook记录供独立检查；运行快照不变。

本地52项初始审核/张量重建/旧连续审核回归通过；扩展21项来源与评测绑定通过，
另1项CPU-only拒绝测试通过，计数有重叠不相加。真实张量和文件执行，外部来源认证/
模型/音频/全量评分在相应单测用替身；完整634正式分支尚无全流程或NPU验收证据。
固定镜像CPU `3a966fc038cd4516928e6b568ae58e4b507f62a0c93129345a0e71c2299c37e5` 已启动，按launch600秒观察；
其快照早于后补CPU-only检查及该1项测试，后续需区分版本。
正式launcher/恢复仍未接通，training_dispatch_allowed=false；学术复现、正式RL收益
及跨任务迁移仍未完成，不把上述工程组件通过作为目标达成。

## 常驻全选择集评测接入准备（2026-11-13 17:38 UTC）

新增ResidentSimpleSelection，按固定0/64/128/192/256/317/384/448/512/576/634步，
对完整9226选择集以73批（72×128+10）执行常驻vLLM推理；完整身份绑定/有限归一化
概率/覆盖通过后才发布指标。gold在推理后本地评分，不进入输入/奖励；原始每批概率
和checkpoint证据hash保存。verify_result独立读取73份原始结果重算全指标，拒伪造汇总。

复用simple_prepared_inputs构造器，原HF/vLLM门槛入口改为委托同一实现，resident复用
已有processor；不是另写转写分支。连续循环评测回调在已接受和提交checkpoint后执行；
每批yield由各rank同步错误/进度，避免非零rank整轮9226评测期间一直等待同一collective。
评测失败或中断毒化owner/session，不静默跳点、不自动重试，不发布部分指标。

本地评测15项、循环12项、共用输入8项通过；之前测试fixture曾因tmpfs存储门槛及错误
schema/依赖构造失败，现仅测试替换外部环境，生产存储门槛保留。模型/音频/engine为替身，
真实张量身份、完整9226概率/计分重算和落盘被执行，不等于NPU结果。
固定镜像CPU `ee061c1abcc0f34cecb2bc1bab3f3743604069b2886333e76e3263bf41dd18b1` 已启动，按launch600秒观察；
前一stage2-initialization CPU仍依自身deadline，未提前观察。
Stage1最近17:31观察213完成、准备214；正式启动器、恢复/独立训练审核接入仍缺，
training_dispatch_allowed=false，已有运行快照不变。

## 正式Stage2阶段切换实现准备（2026-11-13 17:30 UTC）

新增 `simple_stage2_initialization`：要求最终Stage1累计634步及全9226选择集HF/vLLM
独立概率验收，绑定四分片/导出/运行配置和成功终态。共享FSDP恢复器保留精确FP32
master，验证来源Adam后清空状态及残留梯度；RL新日程从0开始，参考固定为最终Stage1。
明确拒绝CE两步前缀、RL工程fixture、部分选择集、旧游标及未清空优化器。

`probe_simple_sampled_npu`增加共享加载分支，`probe_simple_resident_train.continuous`
复用已有循环支持0→634，不复制trainer。原工程fixture仍按1→3执行；运行中快照不改。
正式分支写stage=`simple_stage2_rl`，现有工程恢复器不静默接纳；后续必须显式接正式
启动/评测/恢复与独立终态审核。当前尚无正式启动器，training_dispatch_allowed仍false。

49项本地相关测试通过，包含实际张量的首个新Adam更新精确对照、真实审核器全9226
合成输入概率复算和末条损坏拒绝；模型/发布输入为替身，不是NPU阶段切换验收。
固定镜像CPU容器 `2abd7c8b2429f62e3f4a8350acb30a10f0f73d35166ae0f2e552a9b80258720f`
已启动，按stage2-initialization-cpu-launch至少600秒观察。归档时出现远端时钟落后约
36秒的tar mtime提示，任务已成功启动；身份检查继续使用内容hash，不据mtime重启。

## 连续恢复CPU验收与发布范围（2026-11-13 17:22 UTC）

固定镜像CPU容器 `046ef8c8…` 到期观察exit0：83测试通过，归档的1336份源码逐hash
与远端一致。包含恢复来源/独立审核、精确状态和resident启动/RPC/存储回归；外部
模型/音频/SSH/Docker使用替身。快照早于后续2项输入清单绑定及7项恢复启动器测试，
不混记验收范围。证据 `.local/simple-final-v1/continuous-resume-cpu-verified.json`。

新增恢复启动器7项本地入口测试全部通过，核已有句柄/目录拒绝、卡占用、来源失败、
独立验收缺失、hash变化及成功路径的源码/原审核路径绑定；已接本地观察、独立审核
和收集入口。尚无真实常驻连续checkpoint，不派发恢复，不将CPU结果升级为NPU验收。
Stage1在17:20 UTC观察仍running，四rank正在第207更新（前206步完成），目标634。
正式Stage2仍禁止派发；下一缺口限定为常驻连续更新/恢复、最终Stage1选择集评测和
全量RL统一入口，不新增复杂任务或PPO作为发布前置项。

## 第二更新概率独立验收通过（2026-11-13 17:15 UTC）

finalizer `89be746c…` 已到期exit0并收集，独立复算更新2的HF/vLLM全256输入：
最大logp差9.5367431640625e-7，argmax变化0；更新后resident128对HF差0、对vLLM
最大9.536743e-7、argmax0。原始checkpoint/CPU PEFT/master证据同时绑定且文件hash复查。
证据：`.local/simple-final-v1/sampled-npu-grpo_v1-continue-parity-finalize-verified.json`。

两次分进程RL更新的行为/固定参考/Adam/新策略概率衔接已验收，仍不是常驻连续训练。
真实连续NPU2/3更新及其新进程恢复、最终Stage1/9226选择集和正式全量RL尚待完成。
现有Stage1继续原配置，普通观察不短于600秒。恢复CPU046ef8c8尚待首次到期观察。

## 连续恢复入口与审核准备（2026-11-13 17:13 UTC）

新增simple_continuous_resume在模型加载前绑定独立审核终态、训练终态、四分片/导出、
FP32 master/Adam、算法/参考/完整日程及next cursor；新增probe_simple_continuous_restore
真实DP4重载入口与verify_simple_continuous_restore独立CPU逐元素状态/128概率审核。
精确比较器只在显式continuous=True及正确stage/lineage时支持3–634，旧1/2默认边界保留。
31来源/状态测试、16独立恢复审核测试通过；模型/音频为替身，实际NPU入口仅语法检查。
当前还没有真实常驻连续checkpoint，不能称已恢复或正式RL准入。

resident-launch-cpu0b605fc7到期exit1：34通过/1失败，失败为RPC测试tmp_path过长，
没有进入真实服务/训练。测试socket改用短/tmp专用目录并finalizer清理，特意长basetemp
运行9项RPC测试通过；原失败保留。新continuous-resume-cpu固定镜像任务已启动，
包括恢复审核和联合启动/RPC/存储回归；按launch至少600秒观察，不重启旧任务。
Stage1最新17:10观察完成199/634，准备200。第二更新概率独立finalizer仍待到期。

## 第二更新概率任务完成，独立复算启动（2026-11-13 17:05 UTC）

`0809792d…` 已按十分钟到期exit0，62固定镜像测试通过，HF/vLLM各256输入完成。
独立finalizer `89be746c4a7e152d7d80212b41577b6759e9e6cda1c22d770c78deab01365b11`
已启动，依continue-parity-finalize-launch至少600秒，暂不声称独立概率验收完成。

联合启动器新增成功路径测试，实际执行service/trainer两个入口分支，以模拟SSH/Docker
验证五卡隔离、共享IPC、只读模型数据、服务/观察配置hash绑定与任意cwd源码归档。
共5入口测试通过，仍不等于真实五卡执行。已启动resident-launch-cpu快照早于成功路径
测试与cwd改动，验收范围单独记录。下一步实际联合运行仍等待原Stage1释放卡；
最终Stage1/selection、常驻连续更新/新进程恢复、学术复现与效果迁移仍未完成。

## 常驻两批独立验收（2026-11-13 17:02 UTC）

resident-batch-finalizer `1453ee93…` 到期exit0并收集，真实vLLM两批128、固定参考/行为
共四份原始分布独立复算与已接受输出差0；完整日程人口与G16抽样回放通过。
这证明常驻批次采集，不等于同一actor连续更新。证据
`.local/simple-final-v1/sampled-npu-grpo_v1-resident-batch-finalize-verified.json`。
联合启动器现具备该前置凭证，但还需五卡空闲及实际联合执行。第二更新概率仍按其launch等待。
启动器归档路径改为从repository解析，避免从其他cwd运行时漏收源码；4本地前置测试再通过，
已启动resident-launch-cpu快照早于这一仅路径解析改动，保留版本边界。

## 常驻联合启动准备（2026-11-13 17:01 UTC）

固定镜像连续审核测试39d4fda7…已到期exit0，54测试通过，1327源码文件与派发归档
逐hash一致；证据resident-audit-cpu-verified.json。这是CPU合成审核，外部模型/音频
加载为替身，不是实际连续NPU验收。

新增可归档入口 `tools/jev/launch_simple_resident_train.py`，service/trainer显式两阶段：
先验证独立switch/restore/resident-batch凭证及五卡空闲，再启动NPU4 TP1；训练启动前
核实际docker inspect、镜像、设备、只读共享挂载、同一IPC及ready/config。准备0–3 DP4
恢复原RL step1、连续执行2/3；不抢占Stage1、不因超时重启、不覆盖已有root/handle。
输入与导出经共享只读base可见，IPC单独绑定本地/jev-ipc；每容器cache/source隔离。
31本地service/RPC/storage测试、4实际launcher前置失败测试通过；CPU固定镜像新任务
resident-launch-cpu已启动，按launch600秒；尚未启动真实NPU服务或联合训练。

Stage1最新17:00观察完成192/634，准备193。实际连续运行、终态新进程恢复、最终
Stage1参考及正式全量RL仍未完成；算法/数据/现运行快照不改。

## 第二次更新独立验收通过（2026-11-13 16:54 UTC）

第二次更新审核 `7bd5bd23…` 已exit0并收集：固定镜像53测试通过，四rank/112张量、
精确master重建、Adam1→2、PEFT重载、完整128音频采样回放与固定参考通过，
7923证据文件hash复查一致。更新adapter为
`4c4fdb36028f84921fa4e47dc46d3d5fe2ec93f79c895fca2540f708f52644a4`。
这确认两次分进程工程更新的状态衔接，不是同一常驻服务的连续训练或效果提升。

收集时发现safetensors输出mode0600、root所有，主机普通账号无法读取。仅对已终态
第二次更新的单个模型文件改为0644，前后SHA256与独立审核绑定一致，权限收据
`.local/simple-final-v1/sampled-npu-grpo_v1-continue-export-permission-receipt.json`。
`seal_adapter`现于同步前显式fchmod0644；5项本地持久提交测试通过，包括字节不变和
权限先于fsync。没有修改运行快照或禁用fsync；该新源码仍需固定镜像覆盖。

常驻两批验证 `2b754be4…` 已到期exit0；独立finalizer
`1453ee93070166cdb85e6cae7115e4f72bea9c779ee34aefa7950cb89cb1ff16` 已启动。
第二次更新策略HF/vLLM256验证
`0809792d65ca591b66c907e7517e1d4bcc107eaf0b15e94aa2f480bfc43b925f` 已启动NPU0。
均未观察终态，依各自launch至少600秒；正式Stage2仍false。
Stage1最新16:50观察完成183/634，正在184，原配置继续。

## 连续审核补齐（2026-11-13 16:49 UTC）

`verify_simple_resident_train.py`新增独立完整日程采样复算、四rank master/Adam与BF16
导出重建、post-update128全候选概率、各rank下一游标/日程/固定参考核对；终态实际
CPU PEFT重载检查已接入。初始checkpoint也纳入证据hash。缺失/错误commit拒绝。
新测试真实读写四分片张量、执行G16和身份/概率比较；音频/模型/PEFT外部加载为替身。
19个审核用例加10个覆盖/概率用例共29通过，包括刷新commit hash后篡改master或
回退Adam、错参考/日程/游标、重复slot、概率偏差、非有限/零梯度和错误PEFT加载拒绝。
这是本地测试，不能称实际模型重载或NPU连续验收。凭证见
`.local/simple-final-v1/resident-train-audit-local-tests.json`。

独立审核启动器/收集器新增显式sampled-resident模式，尚未派发连续训练。
固定镜像CPU测试容器 `39d4fda7b5fd1d6bf327ca6314327948c1b6b2b6197b195399313f0472ca82a1`
已启动，首次观察按resident-audit-cpu-launch至少600秒。常驻服务与DP4联合启动器、
真实连续运行、终态新进程恢复及最终Stage1初始化仍待完成，正式RL保持false。

## 发布收口与最新证据（2026-11-13 16:42 UTC）

任务定义已冻结为全量 clean SLURP 单 Final60；正式 Stage2 发布仍未完成。
此次只补公共基座已有验收缺口，不增加新算法、复杂任务或吞吐候选作为前置项。
运行中的 Stage1 快照不改；175 卡4–7最新到期观察完成175/634，正在准备第176批。

固定镜像连续 CPU 循环已通过：28项测试，原生 verl GRPO/RLOO、真实 Gloo DP4、
micro1/2各3次更新、独立梯度/master/Adam对照及第二步恢复。40份产物hash与
1323份源码文件匹配派发归档；这是合成CPU范围，不能代替真实音频NPU连续循环。
证据：`.local/simple-final-v1/sampled-loop-cpu-v2-{collected,source-check}.json`。

第二次完整128音频GRPO更新容器 `eaf12481…` 已到期确认exit0，独立审核
`7bd5bd23…` 已启动；未把容器成功等同检查点/新策略概率验收。
常驻引擎两批输入验证 `2b754be4…` 已启动NPU0，尚无结果。二者依各自launch
至少600秒观察。最终Stage1/selection及真实常驻DP4+TP1连续更新仍待完成。

发布只设一个公共版本，算法作为配置项：GRPO先验收，RLOO-clipped随后复用，
PPO-critic独立预留。每项标明“设计锁定/实现验证/正式可派发”，不将配置存在当作验收。
发布清单固定源码hash、镜像、模型、数据、模板/类别映射、输入处理、精度与LoRA、
优化器/日程、奖励/归约、评测与恢复入口及验收凭证。修复生成新快照，旧曲线保留版本边界。

复杂模式先冻结Q语义、监督、动作分布和Final指标，再接同一工程。Direct、Q-input、
Q-credit分开比较；后两者保持相同输入/输出支持与CE/KL，仅改变过程奖励。
标准GRPO/RLOO先不改；固定过程预算且缺标mask，不逐节点额外标准化，不沿Span序号用GAE。
若实测出现Final信用稀释，再独立注册条件信用变体，并核对期望梯度、方差及Final曲线；
不能将变体悄悄替换标准基线。过程奖励预算不等于梯度预算，联合裁剪与节点裁剪也不等价。

## 独立切换/恢复通过，第二次更新启动（2026-11-13 16:30 UTC）

恢复审核6c0e7945…exit0并收集，四rank FP32 master/Adam精确、全256 FSDP/HF/vLLM
最大logp9.536743e-7/argmax0；7889文件hash复查。切换finalizer-v2 95eb24f0…exit0，
old/new/old各256、精确切回、无HF输入行为256精确相等独立复算通过；6555文件hash通过。
两文件集合有重叠，不能合计成唯一文件数。旧v1写出失败记录保持不变。

第二次实际GRPO更新已启动0–3卡：
`eaf12481a8bdbb59e19290e21189c4d65ddd71ef4e4fdda1164fff0ec1b76ed7`。
使用完整第二批128音频、更新行为、固定原参考、Adam1→2和统一CE游标；尚无结果，
依sampled-npu-grpo_v1-continue-launch.json至少600秒观察。非正式Stage2。

新增本地Unix socket常驻服务/客户端及连续NPU回调入口。JSON控制、SHA256绑定的
weights_only张量传输、label-free renderer、严格序号、异常不重试；临时张量成功消费
后删除，失败保留。46本地socket/采集/循环测试通过（真实socket、engine替身）。
`probe_simple_resident_train.py`准备复用同一DP4加载/Adam、独立TP1服务、更新后全128
actor-vLLM对齐及持久checkpoint，未派发、仅语法检查，不能称NPU验收。
需独立空闲推理卡；当前Stage1占4–7、工程更新占0–3，不抢占运行任务。

连续循环CPU v1在phase0的CPU Adam测试触发torch_npu初始化退出，20通过/5失败，
未进入native DP探针。纯CPU测试显式TORCH_DEVICE_BACKEND_AUTOLOAD=0与非fused/foreach
Adam，新v2 `d3bb859f15e1e6392648f779503950edaaa0bab7fd08cbdc3dbab3bd2040026c`已启动；
native verl阶段保留原镜像环境，尚无结果。原快照不改，依launch至少600秒。

Stage1最新16:29观察完成165/634，正在166。正式全量RL仍未放行；下一独立审核第二次
更新与新概率、常驻批次实际运行、连续NPU回调验收和最终Stage1/selection仍待完成。

## 连续循环与恢复执行（2026-11-13 16:19 UTC）

实际DP4新进程恢复dfabd18d…按600秒到期确认exit0，四rank×64恢复概率对HF差0。
独立CPU checkpoint审核已启动`6c0e79455f774732682ef9d14565ed599b4d0bd54783e3b2aa411284c0223509`，
尚未取得独立状态验收。switch-finalize-v1 041b7686…exit1：32固定镜像测试通过，
verify函数返回后写出run/verified.tmp因父目录未创建失败。保留旧失败，补CLI mkdir；
新CPU finalizer-v2 `95eb24f0e679ba1e57eac86fc68cfb85191ed7570328afeeff0876a53e7f24a9`
已启动，复用原NPU成功证据，不重跑NPU；下游GRPO gate明确指向v2。

新增`simple_rl_loop.run`驱动同一个已加载的session：全量日程、采样→原生actor更新→
不可变导出→独立验收→checkpoint/记录；失败禁止隐式重放、每epoch边界和终态保存，
各rank核对日程/游标内容。28本地CPU生命周期测试通过，包含真实小张量Adam续接；
23切换/批次审核测试通过。尚无该循环的真实NPU回调整体验收，不冒称完整trainer。

固定镜像实际verl四Gloo进程GRPO/RLOO、G16、CE/KL、micro1/2、三次更新与Adam恢复
探针已启动`ff33abf7c0276f1d91cf166789e813f2d2c63e0a26b3425361f091ee405aaa86`。
该不可变快照含最初25生命周期测试，不含后加日程内容一致性/末epoch测试，未观察结果。
源码archive清单在.local/simple-final-v1/sampled-loop-cpu-source-manifest.json。

Stage1最新16:19观察完成156/634，正在第157批。继续按各launch至少600秒观察，
优先独立switch/restore收口→第二次真实更新；常驻批次探针已准备待资源，不派发正式RL。

## 驻留切换成功及全量批次接口（2026-11-13 16:09 UTC）

按600秒检查点确认switch 18ed559b…exit0，原始结果旧→新→旧各256且切回输出精确一致；
第四次无独立HF输入的更新策略行为256也精确一致。独立CPU finalizer已启动：
`041b7686e89b0430a07966891524d23ca0afa96bb193f966342b4f56600bbb5c`，尚未观察结果。
实际DP4新进程恢复已启动0–3卡：
`dfabd18d6f44e3931b96cddbb3585336f25f56b368ef43387e27421f2be7d6e5`，尚未观察结果。
按各自launch凭证至少600秒观察；不得提前声明独立切换/恢复通过。

新增`simple_resident_batch.ResidentSimpleBatch`：完整128日程每批音频仅编译一次，
只把不含标签的renderer传入同一vLLM engine，固定参考与更新行为显式分开；初始两者
同identity时复用同一分布，允许连续FP32更新导出相同BF16字节。保持尾批65真+63零权
padding、固定参考配置副本、精确连续游标；失败禁止重试。采样仍复用actor_batch。

实际两批NPU探针和独立审核入口已准备，原始每次rollout与采样结果保留供重建；
使用现有启动器`--resident-batch`，依赖独立switch准入，尚未派发。
50 transport相关、50 schedule/switch相关以及17新collector/审核测试通过（有重叠，
不可相加为唯一测试数）；音频和engine用替身，非真实NPU或全量trainer验收。

Stage1最新16:09观察完成147/634，正在第148批。正式Stage2保持关闭；剩余恢复、
连续第二次更新、常驻完整训练循环和最终Stage1/selection必须继续完成。

## 更新策略独立验收通过（2026-11-13 15:57 UTC）

按600秒检查点确认finalizer 5dc33a11…exit0并重新核对证据hash：HF/vLLM完整256，
最大logprob差9.5367431640625e-7、argmax变化0；128驻留FSDP对HF差0，对vLLM同上述界。
概率报告5234文件、下一行为报告6545文件hash均通过（两集合有重叠，不当作独立文件总数）。
第二批128音频采用更新行为/固定原参考、游标1及原生DataProto/优势传输通过；
全40513人口/81026使用/634更新与权重复核通过，尚未执行第二次optimizer更新。

驻留旧→新→旧及无重复HF行为采集验证已启动NPU0：
`18ed559b6c03691757e9edeebb0854f9d3e97644598a58aa96dc97e7e07d8cea`。
后续观察遵守sampled-npu-grpo_v1-switch-launch.json的600秒下限。
新进程恢复、第二次更新及连续全量trainer仍待完成；正式Stage2准入false。

## 当前执行契约（2026-11-13，覆盖下方历史状态）

用户再次要求任务定义→工程与配置锁定→训练。本轮收敛为一套公共runtime、两个主算法配置，
第三算法仅保留规约。已有4–7卡Stage1按原不可变快照继续；不追加新全量训练，
不把运行期间修改的工作树视为已发布基线。吞吐候选与正式曲线隔离。

### 已冻结的简单任务

直接音频+固定60类任务定义→一个Final意图；不使用转写串联、文本CoT或槽位值生成。
使用official-clean-intent-v1完整40,513 learning，按来源组隔离9,226 selection；
两者合计49,739条，selection不能回流优化。已知缺陷仍排除，TEST不参与选择。
类别权重为全learning统计的sqrt inverse frequency/cap4/人口均值1，损失除以真实请求数。
固定完整selection的accuracy、60类macro F1、supported macro recall、每类召回和失败率。

### 一套运行基座，分层锁定

公共环境：镜像52ba94e1…、Qwen3-ASR-0.6B、BF16语言q/v LoRA r4 alpha8 dropout0、
FP32 master/Adam，verl原生DP actor/FSDP与vLLM TP1，固定短音频概率对齐profile。
本地/data热写、共享只读。175全0–7已授权；当前Stage1固定4–7，不临时扩成DP8。

现运行Stage1配置：global128、micro1、累积32、LR1e-4、两轮634更新、seed317。
原两步前缀checkpoint/CPU重载/HF-vLLM/驻留恢复及全量processor已验收；完整634终态
与最终selection效果尚未取得。后续状态以engineering状态页和原始凭证为准。
当前全量训练的目标是CE；独立GRPO工程探针已完成一次更新，不等于全量Stage2运行。

吞吐候选：global256、micro1/2（4仅后续候选）、完整人口与顺序不变，每轮159更新。
其中global128/micro2两步NPU吞吐及独立checkpoint审核已通过，较micro1约+17.5%；
新策略概率与全量稳定性仍未验收，不能直接替换现有曲线。增加分阶段耗时和allocated/reserved峰值；候选可把冻结权重
全量hash检查移到checkpoint边界。原配置默认仍每步检查，不更改运行快照。

配置变更必须给出：问题/单项差异、同样本吞吐与显存、全候选概率/梯度/恢复检查、
新配置和源码hash、明确曲线边界。通用微批接口增加不是已证明吞吐提升。

### RL配置与真实实现状态

| 配置 | 固定方法 | 角色 | 当前状态 |
| --- | --- | --- | --- |
| grpo_v1 | 同query G16有放回采样，组均值/样本标准差优势，原生clipped surrogate | 主线 | 固定镜像CPU连续循环通过；NPU首更新/新策略概率/切换/恢复独立通过；两次更新检查点/概率通过；常驻生命周期待验收 |
| rloo_v1 | leave-one-out优势，相同clipped surrogate | 算法配置比较 | 固定镜像CPU通过；尚无NPU更新；明确称RLOO-clipped |
| ppo_critic_v1 | 单步V(x)基线+value loss | 预留 | 未接入验收，不计为已可运行算法 |

共同登记的简单损失（正式发布仍受工程准入约束）：L = sum_real w_x [mean_G L_clip + lambda_t CE + 0.01 KL(pi||pi_stage1)] / Nreal。
奖励为Final正确0/1；process reward=0；clip=.2、固定verl dual-clip=3；每行为批次一次更新；
CE cosine 1→.1，前80%预算衰减；KL参考固定已验收Stage1终态。恒奖励组保留，padding零权。
这属于GRPO/RLOO的明确混合CE/KL配方，不冒称纯RL论文原样复现。
固定镜像真实音频256条vLLM输出回放到native DataProto通过；2026-11-13 15:12 UTC
到期确认GRPO v2完成一个完整128音频/G16/DP4更新，原始恢复256概率差0。此探针
行为来自已验收的两步CE前缀artifact，不能称为最终Stage1参考或连续新行为RL训练。
旧exact_marginal_legacy仅历史路径，不作为主线算法，也不把新结果归到旧精确边缘PPO。
主配置为configs/experiments/slurp-simple-sampled-rl-v1.json；旧design-v2是复杂任务历史候选。

RL实验先锁主线GRPO并观察相同selection上的持续提升；RLOO作为同预算配置比较。
不做RL与SFT胜负比较。两者共用一次工程验收和明确算法差异检查，避免复制trainer。

### Step-wise：先任务后奖励与信用分配

当前Q1/Span1、Q2/Span2、Final/Span3是一次前向同时判决，没有实际采样动作后的状态转移。
因此不沿Q编号使用时序GAE或虚构中间环境状态。问题定义必须先固定为跨数据集语义规约，
不能按真实样本的gold定制Q。缺失过程监督需mask，不自动负例，不要求无监督任务硬凑链路。

首轮保留标准GRPO/RLOO估计器，只比较同输入同输出支持的两臂：
1. Q-input：只奖励Final，各Q提供固定语义规约，过程监督仅按预登记CE使用；
2. Q-credit：相同输入/支持/CE/KL，加可靠过程标签奖励，固定过程预算，缺失mask不重新放大剩余项。
Direct作为整体结构对照，不能把其与Q-credit差异全部归为过程奖励。

一组同时采样的Span赋值是完整动作，联合logprob为各Span之和；联合ratio与各节点ratio
的裁剪目标不同，不可无标记互换。过程奖励经GRPO归一化可能放大，独立节点标准化更会
改变权重。因此先记录Final/过程收益、恒奖励组占比、clip fraction、按任务族KL与梯度；
以Final选择集提升为主要结果，不用总Reward上升代替。若发现信用稀释，再注册节点级
条件基线/分族信用变体，验证期望梯度和估计方差，不修改通用基线名称。

### 最新执行证据（2026-11-13 15:47 UTC）

更新policy的parity-v1因IPC路径过长退出，HF256完成、vLLM0输出；独立v2采用本地
cache短bind后exit0，62固定镜像测试、HF/vLLM各256，记录maxlogp9.536743e-7/
argmax0变化。独立finalizer已启动，尚未验收；不覆盖旧失败，不放宽概率阈值。
完整40513×2日程及60类权重已用实际学习release核验，634游标/末批65真+63padding
接入下一行为与第二次更新。第二次更新仍未派发，需switch/fresh restore先验收。
正式Stage2以及最终Stage1 selection均未完成；学术复现与跨数据集收益仍属原目标待办。

### 最新验收补充（2026-11-13 15:24 UTC）

GRPO checkpoint独立审核已通过：固定镜像93测试、实际CPU PEFT、112张量/四rank
和完整音频采样重放；3917证据文件hash复查一致。新策略HF/vLLM256已启动NPU0，
尚无新策略概率验收。下一批行为版本绑定接入后续finalizer，尚无第二次NPU更新；
完整生命周期及正式Stage2准入仍false，以下15:14发布决策的待办按此更新。

### 当前发布决策（2026-11-13 15:14 UTC）

三个层次独立标识：任务已冻结；Stage1前缀工程证据已接受、全程终态待验；
Stage2算法设计已登记、正式训练发布仍关闭。不能把config中的候选数值或探针exit0
称为完整工程基线已经锁定。机器可读入口仍为simple-final-v1与simple-sampled-rl-v1，
不复制镜像或trainer。旧rl-baseline-design-v2只管尚未准入的复杂任务设计。

当前全量Stage1最新观察是2026-11-13 15:07 UTC：93次更新完成、准备第94批，
目标634次；下次观察遵守原凭证600秒下限。GRPO v2容器faaa4e9f…实际exit0，
收集四rank报告后，独立CPU审核133f8e0e…已启动，尚未取得审核结论。

后续按依赖收口，不再并行扩展任务或训练算法：
1. 独立重建这次RL的112个LoRA张量、四rank master/Adam、真实PEFT加载和G16样本。
2. 同一更新策略完成resident FSDP/HF/vLLM概率对齐，再验old→new→old策略切换及新进程恢复。
3. 在同一runtime验证连续至少两个更新使用新行为版本；现有单次artifact回放不能替代。
4. 完成全量Stage1终态与固定9226 selection评测，锁定正式参考；归档源码/config/数据/模型hash。
5. 上述依赖通过后才发布GRPO全量配置；RLOO复用相同流程且单独登记算法验收。

过程奖励研究在简单基线稳定后开展：Direct→Q-input→Q-credit的解释边界与上节一致；
首轮不改变优势估计器，节点条件信用属于独立命名研究变体。一次前向的多个Span
仅在实际模型/采样支持因子化时使用联合logprob求和，不假设模型消费了前一步的采样答案。

论文定位与本工程实现分开记录：GRPO来源DeepSeekMath §4.1
（arXiv:2402.03300v3），RLOO来源Back to Basics §2.3（arXiv:2402.14740v2），
PPO来源Proximal Policy Optimization Algorithms（arXiv:1707.06347v2）。
本工程额外有CE、精确分类KL、dual-clip和有限动作读出，因此是明确披露的适配配置，
不能声称原论文设置的逐项复现。固定verl源码函数和hash由span_group_advantage核验。

### 收敛决策与发布边界（2026-11-13 14:36 UTC）

本轮复核实际源码：`span_group_advantage.py`调用固定verl原生GRPO/RLOO优势函数，
`span_sampled_hybrid.py`调用原生vanilla clipped surrogate；当前全量作业仍为CE。
锁定一套公共工程，三个算法配置角色；只优先交付GRPO，随后复用相同接口验收RLOO，
PPO-critic不阻塞主线。历史精确边缘PPO不混入新的采样曲线。

简单任务每query从完整60类分布有放回抽G=16个动作，正确奖励1，否则0。
GRPO使用组均值与样本标准差；RLOO使用其余G-1个奖励均值作为基线。
全对/全错组优势为零，保留原分母，CE提供持续监督；记录非恒奖励组比例，
不能只看平均reward判断RL是否贡献有效梯度。G16是有限类别采样，不要求16次音频编码。

**一次更新的解释边界：**当前每行为批次只执行一次optimizer update，更新前actor与
behavior对齐；理想ratio=1，clipping此时不约束第一次参数移动。因而本配置首先比较
优势估计器，稳定性依赖学习率、梯度控制、辅助CE与固定Stage1 KL等已声明项；不能把
原生clipped loss的存在当成有效信赖域保证。如未来采用多次更新复用行为批次，必须
另发配置、明确离策略程度并重新验收，不能静默修改现有曲线。

复杂任务保留同一估计器和整组动作定义。先验证Q-input对Direct的结构收益，再在
相同输入/输出/CE预算下验证Q-credit的额外过程奖励收益。过程奖励预算不等于梯度
预算：GRPO标准差归一化可能放大微小差异。只有观察到信用分配问题后，才考虑独立
命名的节点级条件基线；同时Span乘积分布须符合实际采样因子化，不把固定占位符
当成前一步已采样答案输入。Final、I/S各族指标和冲突率分别报告。

工程冻结交付须包含镜像/依赖、源码文件清单hash、模型和数据hash、模板与标签映射、
音频处理、精度/LoRA、批次日程、优化器、奖励及评测配置、启动命令和验收凭证。
当前工作树不是发布版本；修复形成新快照，并在结果中标明边界。现有Stage1完成后
先验收终态与selection；Stage2必须另过真实NPU行为采样→DP更新→导出→vLLM策略
切换→恢复的一整轮准入。复杂语义任务未冻结前不追加该分支训练。

本轮到期观察：micro2两步探针exit0，256请求为2.21091 req/s，对micro1的1.88157
约提高17.5%；allocated峰值约28.77GiB、reserved约60.06GiB。仅短前缀，不能
外推全量稳定吞吐或直接升级。micro1自身master/导出/PEFT/持久提交审核通过，但
跨运行精确对照退出1：初始adapter文件身份不同。只读追查确认初始权重字节一致，
配置唯一差异为`target_modules`集合的序列化顺序；最终master/Adam跨运行比较尚未
执行。不改历史文件/hash，不绕过失败门槛；后续需区分artifact身份与配置语义等价，
保留原始身份且继续精确张量检查。该问题不构成当前Stage1数值错误证据。

### 下一步顺序

完成当前Stage1→独立634终态状态/重载审计→完整selection基座/Stage1读数；并行只做
隔离的吞吐和新设备工程验证。正式Stage2前必须完成真实音频vLLM行为/固定参考→
DP4更新→版本导出切换→恢复概率一致性，然后冻结唯一配置与源码快照。
复杂任务先冻结Q/监督/指标，再做容量和同输入奖励消融验收，最后训练。

下方均为历史过程记录；其中“未训练”“旧exact PPO”“全局hold”和旧资源状态不代表当前简单主线。


**2026-11-13最新用户调整：当前先交付简单模式工程基线。**
入口为`configs/experiments/slurp-simple-final-v1.json`：direct audio单原生Final60，
clean全40513 learning/9226 selection，固定native绑定；新增I/S审核不阻塞该路径。
用户授权主动调整训练/推理配置，先完成简单任务→共同工程验收→训练，复杂语义实验后置。
下文复杂菜单相关全局hold仅约束该复杂分支，不能阻止当前简单主线。

复用现有instruction入口，新增显式direct_final日程和real_request_mean归约；
两轮各317更新、全量40513录音、尾批63个零权padding，sqrt_cap4权重人口均值1。
本地68测试通过、1项因无verl跳过；已在固定镜像启动CPU测试7de545429d41及
完整processor87671c8b42f1，启动时均running，普通检查至少600秒。输出在175本地
`/data/xingchao/multimodal-jev/simple-final-v1`，共享只读，未挂NPU设备。
这不是训练启动或最终工程通过；状态凭证在`.local/simple-final-v1/*launch.json`。
新增四Gloo进程原生verl更新验收7a52f64eab10，验证真实请求分母、非均匀权重、
显式padding、微批1/2、连续3更新及Adam恢复，含错误分母/padding负对照；
启动时running，尚无结果观察，不提前宣布通过。该探针脚本独立只读挂载，未改运行源码。

2026-11-12，执行用户最新顺序：任务定义和设计→工程搭建、验收和锁定→模型训练。
撤销此前“全量processor就绪后自动恢复Stage1”的推进安排。训练派发保持关闭。
本轮未查询远程实时作业；本文不据历史记录声明当前控制器或NPU占用状态。

**当前执行摘要（以此入口收口，不增加平行基线草案）**

最近一次故障核验：175重启后已恢复连接；175/181均取得同一Ceph同步路径内核异常，
未完成内核修复及本地热存储整体迁移验收。181上次核验仍有其他服务占卡及容量告急，
本轮未查询实时资源，不据历史空闲状态派发。故障和存储限制见第14节及
docs/engineering/host-ceph-crash-20261113.md；早前ready记录中的SSH超时属于历史。

当前语义评测名单已完成：9,226录音/2,146来源组×17面板×三臂=470,526唯一请求，
逐项重建及23依赖pin核验通过，10项相关测试通过。评测v2已接当前名单，旧I_class
生成器仅保历史；每面板Final及等权均值不等于一调用完整输出。见doc109及
semantic-selection-evaluation-design-check-v1；无模型调用、无语义准入，不训练。

固定镜像v2已核实退出1、非OOM、0请求：Transformers显式导入torch_npu时缺少
libascend_hal.so。不再轮询/重启，未执行配对比较，见doc116及
semantic-fixed-image-context-failed-v2.json；早前终态未知记录不再代表当前状态。
资源不可行项已显式加入算法配置的工程发布门槛；三个release gate仍false。

2026-11-13资源硬约束：当前显式eager TP1 attention在30,972 tokens下，仅BF16
scores和FP32 softmax输出就需85.76GiB，超过单卡名义64GiB。semantic-07全部
learning/selection录音均超过这一粗下限，占17面板请求5.88%；DP4/FSDP及累积
不分摊单请求矩阵。源码、全量投影及整数容量核算见doc116/attention-capacity-check，
不是实测OOM。不能将旧短输入eager验收继承到当前长规约；任务冻结后的统一工程
需要内存可承受的attention与新训推概率/生命周期验收，不能只调max_length或
只换rollout kernel。当前不改生产实现或训练配置。

2026-11-13全量输入规模补充：当前规范布局的元数据投影显示Q-input一轮
40,513×17=688,721请求、132.80亿输入token，中位21,217/P95为30,691；音频仅
0.1944%。13/17面板超过16K，长度主要来自规则而非长音频。三臂完整selection
每检查点470,526请求、89.75亿输入token。见doc116及population-projection-check；
3项测试、独立直方图/分位数及两轮历史音频长度核验通过。这不是全量processor/NPU
实测，不选定32K、不更改面板或训练人口。任务冻结需显式预算完整谓词曝光与评测，
工程验收再确定真实吞吐/显存/有效batch；G16不机械计为16次音频编码。

2026-11-13执行收口：算法选择固定为GRPO主线、RLOO-clipped比较、PPO-critic可选，
共用一套runtime及一次typed赋值的action契约；不是三套trainer。现有实现仍是
`span_hybrid_objective → span_finite_credit.verl_loss`的自定义精确边缘PPO，
instruction入口仍为CE；没有已验收的标准GRPO训练曲线。当前任务、工程、实验
配置三个release gate均为false，不能将“算法选择已定”汇报成“工程基线已冻结”。

本轮优先顺序不变：先收口现有语义监督和输入表示，再实现/验收共同runtime与算法，
最后完整Stage1及Stage2。独立人工审核包已就绪但无真实审核者/答案；当前完整输入
CPU探针最大30,972 tokens，所选上下文仍为null。这两个缺口有明确交付物，不能用
继续修改loss或重复两步CE验证替代。正式发布绑定实际工作树源码快照、配置、模型、
数据和验收证据；运行期间不改发布快照，修复另发版本并标明曲线边界。

Step-wise首轮保留标准算法，仅在同输入、同动作支持、同CE/KL下比较Final-only
reward与Final+过程reward；Direct另作整体结构对照。当前并行Span没有采样答案
驱动的状态转移，不沿Q编号套时序GAE。后续条件节点信用是独立变体，不能沿用标准
GRPO标签。过程权重0.025/题只是原始reward预算：GRPO组归一化可能放大它，联合
ratio裁剪也会随输出维数变化；必须同时记录Final表现、分族KL、clip和梯度诊断。
不以Reward单独上升或多个Span正确证明最终决策提升。

固定Omni原生预测另补描述性统计：clean TEST准确率40.08%，来源聚类95%区间
38.48%–41.69%，说话人聚类37.95%–42.22%；59个有样本类别宏召回44.45%。
见[原生基线](60-omni-clean-native-baseline.md)及独立check。没有新推理或训练；
这仍是历史完整JSON协议，不是与新语义Jev匹配的能力比较。

人工独立审核材料已准备：完整128来源/2048请求/15360判决，规则展开及空白表独立
核验；真实CSV证据可接入质量评分，不伪造模型响应。见doc108及human-review-check。
当前审核者未指定、无真实答案；教师语义校准仍未通过，0准入，不训练。

2026-11-13 当前输入诊断已完成，见[doc116](116-semantic-actor-input-context.md)：
17面板的68个真实音频processor请求最大30,972 tokens，56/68超8K、52/68超16K。
全部记录、绑定及271个文件pin独立复核通过；这是独立CPU镜像的输入诊断，
不等于固定训练镜像、全量输入、训推一致性或NPU资源验收。上下文尚未获选，
新增语义监督准入仍为0。先收敛完整输入和监督发布，再验收下述统一工程与算法。

监督通路进一步收口：固定128组的2048请求/15360判决已冻结为实际双消息输入包，
全部来源×面板及user字段独立重建；收集器连接原始执行证据和质量评分，缺失/截断/
格式失败保分母。16项测试包含完整128组文件链路；尚未执行真实质量双审，身份与
预算未发布。见doc108及semantic-quality-inputs-envelope-check-v1。任务全量仍为
40513录音，质量队列不替代训练人口；没有新训练或工程基线发布。
六例JSON外层说明对照已终态：格式6/6但语义参考2/7，5项明确缺席误判未决；该配置不准入完整审核，doc115记录全部失败与分母。下一收口三值语义消费和独立审核，不能靠格式通过宣布监督发布。
三值配对校准已按当前兼容作者参考预登记：I/S各两正两负两未决，12来源、两臂24请求，仅系统判决说明不同。完整问题不按gold缩减；与128组质量审核无来源组重叠，仍非独立gold。运行已终态：两臂各12/12格式完整，参考7/12与8/12，未决仅0/4与1/4；均未通过校准。95配对判决13变，原输出独立复核见doc115。不继续自动提示扫描，不改规则或标签；下一落实独立审核/裁决通路。

重复块共享配对对照也已收尾：相同最长音频的34请求最大28,026 tokens，仍有25/34
超过16K。可逆重建通过，但收益未改变上下文档位，默认表示不变、不继续增加压缩
分支；详见doc116末节。质量评分工具已接入固定128组协议的分母、组bootstrap、
双份审核和裁决覆盖检查；原始响应与规范化标签相互核对。它只计算证据，不认证
审核独立性或自动准入，也尚无真实128组质量结果。新增qwen-plus六例通路诊断也已收尾：HTTP6/6但严格格式3/6，7参考3一致/3失败/1分歧；默认教师不切换，不批量派发。见doc115及独立check。

当前任务依赖修订见[doc109](109-semantic-menu-and-objective-migration.md)与[doc110](110-fixed-capacity-process-reward.md)：actor菜单为65I/55S的17面板候选，Stage1分母已迁移；过程reward采用v3固定容量每题0.025，尾部不加权。[doc111](111-semantic-readout-binding.md)绑定读出布局，[doc114](114-fsc-semantic-transfer-migration.md)完成FSC原生/语义迁移候选。七类依赖设计已有当前候选，仍未获语义准入和运行时验收。下文旧15面板及v2奖励公式仅作历史推导，当前入口以主task/rl配置引用为准。

锁定算法选择与发布顺序：一套共享工程，GRPO为首个验收配置，RLOO-clipped为同底座
比较配置，PPO-critic为单独验收的价值模型参照。三者仍未集成验收；选择已定不等于
正式工程已发布。现有精确边缘PPO保留历史身份，不作为标准GRPO结果。

当前代码身份可直接查`src/multimodal_jev/span_finite_credit.py`：对每个有限分类
Span枚举候选，以旧策略概率求积，优势为
`A_j(a)=w_j*(1[a=y_j]-p_old,j(y_j))`，调用verl的PPO裁剪损失并启用dual-clip3。
它是可加奖励下的自定义边缘surrogate；调用verl损失函数不等于标准GRPO/RLOO
训练器已经落地。历史四卡两步prefix为CE执行验证，不能当作RL效果曲线。
下述标准配置使用完整typed赋值的联合ratio、clip0.2、无dual-clip；差异须明示，
不得沿用旧算法名或将历史曲线与新版曲线拼接。

| 阶段 | 本次交付边界 | 退出条件与当前缺口 |
|---|---|---|
| 任务定义 | SLURP原生60类Final；独立I_present/S_present问句；FSC冻结迁移 | 原生类别与语义本体不要求一一映射。当前65张意图语义候选卡、55类槽位候选；新增语义监督准入0。先完成共同边界、全量监督及正负/缺标政策，再统一菜单、指标和CE/reward。槽位入口v14；绑定关系和新数据集为扩展项。 |
| 工程与配置 | 一个BF16 LoRA/verl/vLLM运行时，算法逐配置验收 | 输入表示定稿后验收全量上下文、驻留actor/重载/vLLM、DP归一化、连续更新/恢复、old/ref同步与评测。固定LR、batch、预算、种子、容差、代码/镜像/模型/数据hash；当前均不能由历史部分验收替代。 |
| 模型训练 | 全量clean SLURP，Stage1消费指令，Stage2 RL+衰减CE | 从同一个已冻结Stage1起点开展各实验臂；只选择已验收配置，不在运行中改Q、奖励、算法或工程快照。 |

任务冻结允许输入自身歧义保留mask，但不允许用mask掩盖未定义的类别边界或监督不足。
原生clean标签与新增语义标签分层；复现原生Final不要求重新认证全部原生标签。
Final使用完整原生类别及描述名，不能把历史候选语义边界冒充其充分必要定义。
过程I_present允许重叠，I_class只保留为另行登记的成员重编码对照，不硬折叠到Final。
当前语义目录入口为semantic-intent-catalog-candidate-v5，slot-consolidated-design-v14；
原生/语义菜单、目标、奖励、读出、评测、上下文和FSC均已迁移为候选设计。
旧60I/15面板配置不再作为当前入口。训练人口保持40,513录音，不能退回小参考集。

Step-wise按以下顺序定位收益，避免一次改变多个因素：

| 比较 | 唯一希望检验的因素 | 必须保持一致或明确披露 |
|---|---|---|
| 固定Stage1起点→Final-only RL曲线 | 混合目标下任务指标随RL阶段更新的变化 | 全量人口、固定评测、衰减CE与KL日程；不宣称纯RL因果收益，不增加RL/SFT胜负比较 |
| Q-input→Q-credit | 可靠过程RL奖励是否改善Final | 相同输入、动作成员、过程CE、KL、G、更新预算及起点，仅过程reward不同 |
| 联合信用→节点条件信用 | 信用估计/局部surrogate是否更有效 | 相同reward与监督；更换ratio/裁剪必须独立命名，不再称标准算法 |
| Direct→带Q | 整体结构的收益与成本 | 输入、动作维度、辅助监督及联合裁剪/KL几何均可能变化，不单独归因于过程奖励 |

首轮不改标准GRPO/RLOO的优势定义。重点记录Final恒定组中的过程放大、联合ratio裁剪、
每族KL与梯度、有效正负监督覆盖和Final指标；过程reward上升本身不构成任务收益。
算法初始数值候选沿用v2，不新增调参轴；学习率、有效batch、总预算等未决字段须在
任务定稿和资源验收后一次补齐，再发布可运行配置。普通实验不触发工程重构；缺陷修复
发布新版本并重验受影响链路，修复前后曲线保留边界。

权威入口：[算法配置v2](../../configs/experiments/rl-baseline-design-v2.json)、
[任务v2](../../configs/tasks/jev-task-design-v2.json)、
[监督计划](../../configs/tasks/slurp/semantic-supervision-plan-v1.json)。
以下章节保留推导、验收证据及历史修订；其中日期和运行状态不能当作实时状态。

**收口交付清单（执行顺序，不是新增启动配置）**

| 顺序 | 一次收口的产物 | 验收证据 |
|---|---|---|
| 1. 任务发布 | 原生Final与独立语义Q的完整定义；统一监督层；固定菜单、均衡、Stage1/Stage2目标、评测和FSC协议 | 全量来源身份及有效正负覆盖；规则兼容、冲突与缺标分开；语义复核与结构检查分开；七份依赖配置完成迁移。不能用未评估标签补负。 |
| 2. 共同工程发布 | 同一个模型/processor/输入编译、BF16 LoRA、DP4、vLLM、checkpoint及评测入口 | 定稿输入全量上下文/资源检查；驻留actor、重载与vLLM全候选概率及决策对齐；连续更新与恢复；old/ref版本绑定；全局batch归一化与DP/累积一致性。容差须先登记。 |
| 3. 算法逐项发布 | 首先GRPO，其次RLOO-clipped；PPO-critic为可选独立验收项 | 同一行为批的优势和loss对固定verl实现；实际梯度/参数更新、保存恢复及概率同步。函数存在或CPU算术测试不替代端到端验收。 |
| 4. 实验配置发布 | 学习率、有效batch、G、更新数、种子、CE衰减、KL、评测点、资源上限全部显式 | 绑定实际代码快照及镜像/模型/数据/任务/奖励/算法身份；不得用未提交工作树的HEAD冒充完整版本。每次运行记录其发布身份。 |
| 5. 正式研究 | 全量Stage1，再由同一Stage1起点开展Stage2；Q-input/Q-credit为首轮过程信用对照 | 原生Final、宏平均/尾类、过程正类召回与误报、监督覆盖、成本和固定检查点曲线；FSC冻结权重及目标配置，不用目标反馈选择模板或checkpoint。 |

任务冻结后，普通实验只切换发布配置中的研究因素，不重构底座。必需缺陷修复另发工程
版本并重验受影响项；改变任务或奖励也另发版本，不把修订前后接成同一训练曲线。
尚未验收的PPO-critic不阻塞已通过的GRPO/RLOO配置，但目前三者均未通过。

**冻结范围与停止扩展规则**

本轮只收口现有任务和配置，不新增任务族、算法分支或训练入口。首版范围为原生
Final60与独立I_present/S_present，绑定和执行状态后置；语义卡仍须完成现有边界和监督
验收，不能用原生类投影替代。65I/55S是本体目录大小，不要求一次输入120个问题。
离线教师的16面板日程也不自动成为actor训练日程；actor输入须按自身上下文和成本验收。

工程只发布一个共同版本，标准算法保留三个命名配置、逐个验收，不维护三份trainer。
同一输入下G=16表示从typed分布采样16份完整赋值；因当前各Span概率不依赖已采样
答案，可复用一次策略前向得到的分布，无须为了这16个动作重复16次音频编码。
这项效率设计需在实现时验证采样独立性、old版本固定和请求归一化，当前未接入。

正式冻结时补齐数值参数和容差，生成包含实际源码的不可变发布快照。实验结果绑定快照，
不能只引用不包含工作树改动的HEAD。任务/Q/奖励发生语义改变时新建实验版本；只有
实现缺陷才修工程版本，并重验受影响项。任一配置的验收失败不通过临时改默认值绕过。

新增语义监督准入仍为0。89请求槽位校准已结束，75完整、14截断空正文；110个参考
目标94返回一致、16失败保分母，见[doc112](112-slot-teacher-calibration-observed.md)。
后续6请求效率提示对照也已结束，4完整、2截断；原3成功的24判决不变、原3失败仅1恢复，
见[doc115](115-teacher-output-probe-observed.md)。已完成意图校准127/130明确目标参考
一致，3失败保分母。这些作者参考对照不是独立准确率或训练准入。教师离线推理与
actor一次Span读出是不同链路；当前通路不释放128组或全人口标注。

本次源码复核仍确认：现有RL调用精确边缘PPO；instruction入口使用CE；三个新算法
profile均为not_integrated_not_accepted。当前瓶颈是完成任务发布和统一工程验收，
不是继续增加规则版本或重跑两步CE。首版保持现有65I/55S范围，集中裁决已登记冲突、
落实完整输出通路及独立审核、准入全量监督、验收当前输入表示；之后才进入工程实现与验收。独立审核
既要检查分歧，也要覆盖一致样本、正负与歧义，不能以同一作者或同一教师的自洽代替。
任务冻结必须预先写明覆盖与质量门槛，不能等观察结果后再决定通过标准。

过程信用首轮同时检查三个风险：GRPO对Final恒定组中的小额过程差异放大；多Span联合
ratio的裁剪随动作维度变化；相近语义Q选择性重复导致奖励权重偏移。固定总cap只约束
原始reward，不保证梯度预算或概念权重不变。重叠审计以准入监督为依据，不能按当前样本
gold动态去重；相关证据见doc53、doc54及doc92。先跑标准算法的同输入奖励对照，再决定
是否增加节点条件信用变体，不把尚未验证的优化写入标准算法配置。

## 1. 现在实际用的是什么

最近NPU两步前缀调用`span_hybrid_actor`，stage=instruction，仅优化分类CE，没有RL。
已实现的task分支使用`span_finite_credit.verl_loss`：

```
A_j(a) = w_j * (1[a=y_j] - p_old,j(y_j))
L = E_{a~p_old,j}[PPO_clipped_surrogate(ratio_j(a), A_j(a))]
L_task = L + lambda(t)*CE
```

期望对每个有限候选精确求和；各Span边缘分别裁剪；调用固定verl的
`compute_policy_loss_vanilla`，clip=.2，dual clip=3，没有critic、GAE或GRPO组标准化。
这是“有限候选精确期望的边缘PPO定制变体”，不能称为标准PPO或GRPO。
它只精确处理当前可加奖励，不能直接宣称精确优化集合F1或组合正确率。
历史小样本RL未形成可信的独立任务提升证据；混合RL+CE仅有真实四rank CPU梯度证据，
不能称本轮全量RL已经运行。

## 2. 统一底座，不为算法各造一套系统

保留一个固定镜像、模型/processor、音频处理、Q编译、typed策略、LoRA/FSDP、vLLM、
数据/监督映射、保存恢复和评测链路。算法配置只切换优势估计、critic及明确登记的loss。
用统一的request/action样本接口，不把候选概率假装成生成token轨迹。

候选底座：本地Qwen3-ASR-0.6B，BF16执行，FP32 master/Adam，语言q/v LoRA r4/alpha8，
175物理卡4–7 FSDP DP4，vLLM TP1。镜像固定`52ba94e1…`，verl commit `e9405d…`。
HF/vLLM修复后256请求一致性已通过；全量恢复、新Q、新算法未全部验收，因此这是
release candidate，不把部分检查通过写成稳定正式基线。

冻结后普通实验仅改变登记配置。真实工程缺陷另建修复分支/发布版本，重跑受影响验收，
曲线保留版本边界；不在运行快照上热修，不把每个实验都变成基础设施重构。

## 3. 登记三种通用算法配置

固定同一完整Span赋值为一个动作。当前策略
`pi(a|x)=product_j pi_j(a_j|x)`，joint logprob为各Span logprob之和。
它是一轮contextual-bandit决策，不是Q1答案驱动Q2的时序环境。

| 配置 | 优势估计 | 价值模型 | 本项目实现边界 |
|---|---|---|---|
| PPO-critic | 一轮原始优势 `R - V(x)`，固定verl再做masked whitening | 需要，仅消费合法音频/任务输入 | 标准clipped PPO；需新增并验收critic生命周期 |
| GRPO | 同输入G个动作，`(R-mean)/std` | 不需要 | 保留组相对优势；使用单一联合动作ratio，明确是typed-action适配 |
| RLOO | `R_i - mean(R_others)` | 不需要 | 登记为RLOO优势＋PPO裁剪，不能冒称无裁剪REINFORCE原实现 |

本地固定镜像源码实际存在三种优势函数及PPO loss，已核验core_algos SHA256
`fadfafa83d94ba40d7ac2abeedb6e99ea0716b42f8744c421ecbb371db8b4d24`。
源码支持不等于我们的音频Span端到端接入完成。
算法出处：PPO [1707.06347](https://arxiv.org/abs/1707.06347)、GRPO/DeepSeekMath
[2402.03300](https://arxiv.org/abs/2402.03300)、RLOO
[2402.14740](https://arxiv.org/abs/2402.14740)。本轮核验的是本地实现，不宣称重新检索所有最新变体。

初始可比较配置登记在[配置文件](../../configs/experiments/rl-baseline-design-v2.json)：
G=16、temperature=1、clip=.2、不用dual clip、每批行为样本2轮优化、相同参考KL及
衰减CE。PPO/GRPO/RLOO使用相同输入、采样预算和joint ratio；critic成本单独计量。
这些是待任务定稿与工程验收的初始值，不是已证明最优或已允许运行的正式锁定配置。
最终发布还须固定LR、有效batch、critic细节、随机种子集合和完整评测预算。

现有精确边缘PPO作为第四个定制诊断配置保留。与采样joint PPO的差异包含枚举/采样、
信用分配和裁剪层级，不能把差值单独归因于“精确优势更好”。需要分因素消融时单独登记。

## 4. Step-wise应该改什么

这里step是语义判决节点，不是默认的时间步。首要调整是奖励、信用、归一化，而不是
把每个Span机械地传给GAE当作连续状态。Final经共享隐藏表示学习；当前离散前置输出
没有被Final读取，不得用RL记号伪造因果依赖。

**先不改优化器。** 三种基线先用相同Final reward，建立可复现的任务曲线。然后在同一个
基线、相同输入上加入有可靠监督的Q信用。当前v3契约为：
`R=R_Final+0.025*sum(mask_I*r_I)+0.025*sum(mask_S*r_S)`。
每族最多4题、奖励上限0.1；尾面板不足4题及缺标均不重新分配预算，空族为0。
不能再用实际题数K作分母；权威契约见
[奖励v3](../../configs/experiments/reward-contract-design-v3.json)。
Final与过程分量分别记录；缺标mask。辅助CE固定并单列，不做RL/SFT胜负对照。

**再改节点信用。** 研究`A_j(a_j)=E[R|x,a_j]-E[R|x]`。可加正确性奖励可精确计算；
绑定或集合关系因子用小范围枚举/预先登记的采样，不能声称穷举大集合。
只有Final奖励且离散动作因子化时，独立过程动作的精确Final信用是0。
给每个过程动作广播同一组优势可能产生采样噪声；条件信用是否改善Final需要实测。

此时若由联合ratio改成节点ratio，就成为新的算法变体，必须另立ID，不在GRPO/PPO/RLOO
基线名称下静默替换。过程reward与CE同时变化的收益也不能只归因于RL信用算法。

**处理尺度与稀疏性。** 不对Final和每个Q分别单位方差标准化后再声称过程总预算仍为0.2。
GRPO总奖励组标准化会改变跨query权重；RLOO保留奖励尺度但方差可能更高。
G=16近均匀60类下全错概率约`(59/60)^16≈76.4%`；记录零方差组，不暗删难组或
自动重采样来抬高数据效率。CE可提供补充梯度，不能将混合目标收益全部称为纯RL收益。
多Q loss按真实query归一化；类别平衡在外层预先定义，不能指望GRPO组内奖励乘常数
在标准化后仍保留类别权重。过程梯度范数/方向与Final同报，奖励预算不等于梯度预算。

## 5. 三个发布门槛

**任务冻结**：I/S/B等Q的真值语义、完整定义表、单/多意图和缺标政策、固定菜单、
Final关系、简单/复杂切片、source划分、reward与CE、指标和迁移协议全部版本化。
现有slot定义仍有draft/缺标，不能把当前intent工程发布扩大为全slot训练发布。

**工程冻结**：上述任务接口下实际processor/上下文、BF16、vLLM概率、四rank梯度和
padding/mask、连续更新/保存恢复、采样复现、critic（仅PPO）、完整评测及资源上界
通过。发布镜像+代码commit+配置+数据hash+验收矩阵。必要的短更新只能作为明确登记的
工程验收，不以此替代或提前开展研究训练。

**实验启动**：在锁定版本上准备Stage1的规约/Q消费训练及独立指标；固定Stage1起点后
运行已验收算法配置和step-wise消融。先报告每种算法自己的持续提升，再判断节点信用
是否提供额外收益；冻结权重和Q形式做FSC等迁移。基线复现和Omni同口径定位仍需完成，
不因新增算法配置取消原目标。

当前三个门槛未全部满足，新训练派发为false。继续可做任务设计、只读调研、已有验收
归档和必要的实现审计；不再以“工程跑通了几步”代替任务已经设计清楚。

## 6. 稳定发布的验收单与变更边界

统一底座只维护一套，共享数据加载、音频输入、typed action、训练循环、保存恢复和
评测。PPO-critic、GRPO和RLOO-clipped是三套配置；最后一个名字明确保留裁剪适配，
不与原论文的无裁剪REINFORCE-RLOO混称。历史实验目录里的同名算法配置不自动获得
新基线资格。当前注册表仍为设计稿，不能直接作为启动参数使用。

| 发布项 | 必须提交的验收证据 | 当前依据与缺口 |
|---|---|---|
| 任务契约 | 定义表、真值/缺标、固定Q菜单、Final关系及监督覆盖 | 原生Final60；独立65意图语义卡与55槽位卡，监督准入尚未完成 |
| 数据日程 | 完整learning人口与hash、来源隔离、类别/定义覆盖、padding有效权重 | clean intent人口40,513；候选日程已审计，语义负例及最终均衡未冻结 |
| 输入一致性 | 冻结音频处理与processor；全日程上下文不截断；训练/rollout请求身份相同 | 当前语义版本独立CPU诊断68请求：最长30972，56超8192、52超16384；32K仅覆盖此次测量，未定稿或获NPU/全量验收。历史短输入不继承 |
| 概率一致性 | 驻留actor、保存重载、vLLM的全部候选与联合概率；同一adapter身份 | 256请求重载HF/vLLM最大logprob差9.54e-7；驻留actor仍缺证据 |
| 更新正确性 | 四rank与参考梯度、mask/padding、优化器状态、连续更新后恢复等价 | 混合目标四rank CPU有证据；三种新算法完整NPU生命周期待验收 |
| 行为策略生命周期 | old/ref/actor身份分离，固定old跨两轮更新，更新后明确同步vLLM | 新算法不得复用无身份约束的旧概率缓存 |
| PPO价值分支 | critic输入无答案泄漏，return/advantage正确，保存恢复及资源预算 | 尚未接入 |
| 评测与运行 | 同一population和完整指标、固定检查点、吞吐/显存/时长、带退出码终态 | 尚无新任务完整验收和正式效果曲线 |

每项通过须绑定代码commit、镜像digest、任务/实验配置hash、数据/模型身份及证据路径。
全量processor历史结果仅覆盖其原任务视图，不继承为新Q菜单的通过证明。
数值阈值应在验收运行前固定，不能根据失败结果放宽；HF/vLLM已有差值是测量结果，
不是自动推导的新验收阈值。

冻结前还需填完：actor/critic学习率与调度、有效batch、训练总预算、固定种子集合、
评测频率、概率容差、资源上限。不得用启动器默认值补齐这些字段。冻结后一般实验
只能选择已登记profile、任务及seed；reward、Q菜单、CE日程、采样/裁剪策略变化均登记
新实验配置。模型/依赖、数据准入、forward、loss、保存恢复的改动发布新工程版本，
重验受影响链路，原版本曲线保留，不拼接成同一条连续训练曲线。

## 7. 首轮实验顺序与step-wise的可证伪目标

完成任务冻结后，先实现并验收GRPO与RLOO-clipped共享无critic路径，PPO价值分支单独
验收；按第9节逐配置发布，未验收的critic不阻塞已验收配置。此排序是减少实现分叉，不预判算法优劣。
工程短更新记录为验收，正式训练始于发布后的Stage1。Stage1学习规约/Q消费，Stage2
使用同一Stage1 checkpoint开展RL+衰减CE；不重启RL/SFT胜负比较。

统一比较单位为真实音频query、完整typed action及全局optimizer update。相同输入的
G个动作可从一次前向得到的categorical分布采样，不默认意味着G次音频编码；实际前向
次数和资源成本单独统计。算法间固定样本预算、CE与KL，critic额外成本明示。

step-wise首先登记以下三个实验臂，再考虑条件信用算法：

1. Direct：同一任务定义，仅Final读出。
2. Q-input：加入固定的可迁移Q菜单，仅Final奖励。
3. Q-credit：与Q-input完全相同输入，加入固定总预算的可靠过程奖励。

Q-input与Q-credit保持同一过程CE设置，因此二者差值才可归因于过程RL信用；若过程CE
同时增加，必须命名为联合辅助监督实验。Direct与Q-input的差值包含结构和计算变化，
报告token与算力，不能仅称奖励算法收益。

任务设计应先说明每个Q提供什么信息。例如I_class若只是Final one-hot的重述，其作用
可能是定义消费和冗余监督，不能预称多步推理；只有S/绑定等有可靠独立标签且与Final
任务有关时，才检验其附加判决价值。多个并行输出答对并不能证明Final使用了这些输出。
当前结构允许共享隐藏表示的辅助学习，尚不具有离散答案到Final的显式因果链。

持续提升以预定检查点的Final指标、尾类表现及来源组配对区间验证，不要求每一步单调；
同报过程指标、零方差组比例、KL、clip fraction、有效监督覆盖和成本。过程reward上升
而Final无收益即为未支持研究假设，不能以累计reward替代任务结论。跨数据集先以冻结
源域权重、目标任务定义和模板评测FSC；目标测试集不参与问句与检查点选择。

## 8. 当前统一设计入口（v2，尚未发布）

[算法配置v2](../../configs/experiments/rl-baseline-design-v2.json)统一引用
[任务v2](../../configs/tasks/jev-task-design-v2.json)、奖励v2和上下文候选；当前完整槽位规约已否定8192可用性，见第11节。
旧v1保留为历史设计；新设计不再继承mean-valid奖励。release_gate逐项列出未完成项，
不以默认参数补齐，不把登记配置当作可启动配置。当前HEAD为7087940，工作树候选
尚未形成冻结发布；本次仅合并设计口径，没有改训练实现或派发任务。


## 9. 收敛决定：版本发布与任务监督分层

本节修订第7节“三种配置通过后发布共同基线”的串行阻塞：维护一套工程底座，
各算法单独验收。首选接入GRPO；RLOO-clipped作为同底座的尺度/优势参照；PPO-critic
作为价值模型参照。首选依据是减少工程分叉，不是已观察到效果更好。共同任务与工程
门槛通过且GRPO验收完成后，可以冻结该配置；尚未验收的critic配置保持不可派发，
无需为了新增critic改动已发布GRPO版本。当前所有新算法仍未验收，不触发训练。

任务冻结也要区分两个监督契约：

- 原生基准：以已发布clean人口及scenario/action为目标，隔离已知omni缺陷；报告
  原生类别指标和固定分母。复现这一契约不要求给全部原生标签重新作语义认证。
- 新语义Q：候选定义、真值政策及覆盖独立发布；I_present、S_present、B_bound不能
  从原生单标签或未出现的标注直接补出负例。现有完整语义层计划用于这一契约。
  若实验将候选定义与native Final一起供给，已记录的不匹配仍须解决或明确mask；
  不能借原生契约绕过定义冲突。

当前104录音/23组是“与项目候选定义存在问题”，不是已证实104条官方标签错误。
候选卡也可能过窄，应先修订任务级定义和优先级，再做规模化监督发布；不继续用
零散人工小集代替全量任务设计。固定官方源码核对见
[来源审计](slurp-definition-authority-v1.json)：官方intent评分使用scenario/action；
评分器跳过缺失预测、预测加载覆盖重复ID，因此必须在评分前检查人口完整且ID唯一。
源码树未发现独立类别边界指南不等于学界不存在该指南；本次PDF获取失败，没有阅读
论文全文。候选定义不能冒充官方定义。

接下来的集中交付按顺序为：任务发布包（定义、监督、菜单、指标、loss/reward），
工程发布包（镜像/代码/数据/模型/配置身份及验收矩阵），实验配置包（LR、batch、
预算、seed、检查点及成本上限）。每包一次性核对未决字段，不以启动器默认值补齐。
冻结后只在登记的配置维度内开展实验；修改任务、算法或工程均保留独立版本边界。

本次只修改设计及审计记录，没有改actor、重标数据或派发模型任务。当前尚不能给出
可信的训练开始时间：主要缺口是新语义监督、驻留actor一致性和标准算法生命周期，
不是再跑几个CE步骤。

## 10. 执行政策与已有验收收拢

2026-11-12复核发现`academic-baseline-lock.json`仍允许新权重训练，且保留
“matched SFT and RL comparison”。已将旧政策完整移入history，当前政策对齐本文件：
任务先行、派发false、Stage2为RL+衰减CE，三个算法逐配置验收。论文记录不变。
该JSON是登记表，不是启动器强制门禁；修改它不等于完成运行时防护或停止远程作业。

原生评分入口已实现人口完整性、重复/未知ID、请求/标签身份检查；解析失败作为
无效预测保留完整分母。本轮确认现有实现，不新增评分器。评测协议中“有效预测覆盖”
改为“预测记录覆盖”，格式有效率单列。针对评分入口、联合裁剪、组信用与奖励契约
运行37项CPU测试全部通过；评分入口测试使用替代metric，不宣称重验官方指标数值或NPU。

Step-wise设计须保留两项已发现的机制风险（docs53/54）：多输出联合ratio会改变裁剪
和KL尺度；GRPO组标准化可放大小额过程reward，原始0.2预算不是梯度预算。
因此先冻结标准算法，再逐项检验过程奖励和条件信用；不把数值toy结果当作模型收益。
本轮未改训练代码、未派发训练，也未将设计配置升级为正式工程发布。

## 11. 2026-11-13 收敛核验：当前不能继承的两个默认值

固定CPU镜像对当前槽位v3/政策v3/边界v4执行90次processor请求，84次超过8192，
最长14710。历史7412上限仅覆盖旧槽位短定义；probe退出0表示测量完成，不表示
上下文通过。已撤回当前上下文的选定状态，未增大运行参数、截断定义或派发模型。
下一步先确定语义输入表示，再验收具体上下文及资源；16384仅为待评估选项。

固定verl的`compute_gae_advantage_return`在计算return后调用`masked_whiten`。
PPO配置现明确使用该标准化，而critic仍回归未标准化return；不能把原来的
“无额外标准化”理解成调用函数返回raw R-V。一次完整行为批上计算有效动作统计，
再分发rank/切microbatch；分布式一致性尚待验收。GRPO和RLOO定义不变。

本轮交付和有序剩余工作见[收敛决定](61-baseline-convergence-and-context-gap.md)。
所有新算法仍为not_integrated_not_accepted，training_dispatch_allowed=false。

## 12. 当前交付边界：停止扩散，按三个包收口

再次核对实际源码与v2配置：`span_finite_credit.verl_loss`仍为精确边缘PPO；
`train_slurp_instruction_npu.py`当前调用`stage='instruction'`，不能将其两步CE验收
汇报成GRPO训练。当前HEAD 7087940及未提交设计/实现草稿不构成冻结工程版本。

本轮收口不再增加算法或任务族：主配置GRPO、比较配置RLOO-clipped、可选PPO-critic，
统一工程只维护一个实现入口。这里锁定的是算法选择与执行顺序，完整数值配置尚未冻结。

| 顺序 | 本轮明确的交付范围 | 完成标准 |
|---|---|---|
| 任务包 | SLURP原生意图Final；固定I/S谓词；语义标签和缺标政策；Direct/Q-input/Q-credit；冻结FSC迁移协议 | 完成剩余属性/关系/定义目标边界和意图映射，合并一版55类规约，迁移并核验全量监督，冻结菜单、指标、奖励与CE |
| 工程包 | 现有BF16 LoRA/FSDP/vLLM候选；GRPO/RLOO共用训练循环；保存恢复及评测 | 冻结输入表示后全量上下文/资源通过，驻留actor→重载→vLLM一致，连续更新/恢复和old/ref同步通过，各profile独立验收 |
| 实验包 | Stage1指令消费，固定其checkpoint；Stage2 RL+衰减CE；过程信用对照；FSC冻结迁移 | 明写LR、有效batch、G、更新数、种子、评测点、容差和成本上限；绑定代码/镜像/数据/任务/算法hash后才派发 |

当前槽位入口为`slot-consolidated-design-v13.json`（类型v15/政策v15/边界v16）；
监督进度以`semantic-supervision-plan-v1.json`的current_workpack及doc85为准。
旧完整规约max14710、共享规则表示max11721均来自90请求CPU探针，后者83请求超8192；
二者均不是最新规约的全量上下文验收。不能继承旧测量宣布新版通过，也不能只增大
max length就宣布四卡训练可用。语义建议、结构绑定核验和独立语义复核分开记录。

Step-wise首轮只改变可靠过程reward；标准GRPO/RLOO定义不改。后续条件节点信用
独立命名、独立验收。在当前因子化策略中，Final不消费采样出的前置离散答案；前置
位置的隐藏表示可以参与Final计算，但需要实验验证，不能把多个正确输出称为已证明
的因果推理链。I_class也可能只是Final标签重编码，应与独立语义监督分开解释。

typed策略一次前向给出各节点概率，G=16可据此采样16组完整动作，不必重复16次音频
编码。应在接入时复用概率并精确实现样本加权，报告真实前向/反向成本；不改变组优势、
动作成员或联合裁剪，也不声称现有训练器已经完成这项优化。

正式研究结果以Final曲线、宏平均/尾类、过程正类召回/误报、有效监督覆盖和冻结迁移
表现为准。奖励上升而Final无改善，不支持过程信用有效；多Q与Direct的差别同时包含
输入、辅助CE和联合裁剪/KL几何，只有同输入Q-input/Q-credit才隔离过程RL奖励。
本次没有启动模型任务、修改生产训练实现或发布稳定版本。

## 13. 本轮复核：固定范围和停止条件

继续使用本文件与`rl-baseline-design-v2.json`作为唯一收口入口，不另起工程或算法设计版本。
这里确认的是发布范围与顺序，不能将尚未验收的设计配置改称稳定工程。

当前任务包包含60个原生意图类别和55个候选槽位谓词。clean learning有40,513录音，
但新增语义Q的正式准入仍为0；current-rules-v12中的15个当前明确坐标是参考建议，
不是可训练标签。因此任务包仍在第一阶段，不能通过换算法绕过监督缺口。

任务设计收口只覆盖SLURP意图Final、固定I/S问句及FSC冻结迁移协议；绑定关系、
其他数据集和新算法保持扩展项。公开问题按共同定义边界集中解决，输入自身歧义保留mask。
任务版本冻结必须同时绑定监督准入、有效正负覆盖、固定菜单、CE/reward和指标；
原生clean标签与新增语义标签分开发布，不把全量原生意图再认证当作必要前提，
也不把新增语义标签的审核要求绕过去。全量训练人口不缩成手工参考小集。

工程阶段只交付一套共享runtime，首先验收GRPO，然后以相同数据/action/训练循环
验收RLOO-clipped；PPO-critic另行验收，不能拖动已冻结的无critic版本。
冻结标识须覆盖未提交实现的最终代码快照，不以当前HEAD代表整个脏工作树。
常规实验只改变已登记配置；缺陷修复需要新工程版本、影响说明及对应回归证据。

首轮研究依次为：固定Stage1起点；同一算法的Final-only曲线；同输入同CE/KL的
Q-input与Q-credit对照；最后才评估节点条件信用。Direct用于整体任务结构对照，
因输入、动作维度和联合裁剪均不同，不能单独证明过程奖励有效。
训练必须报告原生Final、宏平均/尾类、过程正类召回与误报、监督覆盖，以及吞吐和成本；
只有reward上升不算主任务改善，训练过程也不要求每一步指标单调上升。

本轮仅复核源码及已有记录，未查询远程实时作业、未重跑NPU验收、未启动训练。
本文件中的NPU结论是历史证据，不表示当前卡上运行状态。


## 14. 主机Ceph故障纳入工程门槛

175/181均取得同一fsync→Ceph空指针异常栈，模块身份一致，见
[故障记录](../engineering/host-ceph-crash-20261113.md)。换镜像不能修复宿主机模块。
新运行遵守[本地持久化规约](../engineering/runtime-storage-policy.md)：保留本地fsync，
Ceph只读消费、健康客户端归档；不以删除fsync换取表面稳定。instruction入口已加写路径
与容量检查，历史快照不变。内核兼容修复、完整热路径隔离及归档恢复验收加入rl-v2工程
门槛，三项release gate仍false；本次故障处理不改变任务定义或提前派发训练。

2026-11-13任务指标收口：evaluation-v2/FSC-v3现明确原生固定类macro F1、
supported macro recall、语义正负与无效分母、缺标null、完整槽位面板及ragged面板
聚合，详见doc109计分分母节。7项算术测试及手算复核通过，selection名单hash未变；
历史评分器不继承为新语义scorer，独立实现准入仍缺失。原固定镜像v2已失败终态：
processor导入显式加载torch_npu缺libascend_hal.so，0请求，不再轮询/重启，见doc116。

终态计分契约进一步明确：固定全请求收齐后评分，显式失败保分母，缺失/未完成拒绝；
原生Final与布尔Span独立映射。完整470526合成终态及18360实际布局符号组合核验，
23项算术/账本测试通过，见doc109。仅任务计分参考，不是生产scorer或模型结果。
人工审核包再次只读核实15360空项、身份未指定；不虚造审核或扩大教师调用。

学术基线补证：SpeechBrain v0.5.11固定SHA已有SLURP HuBERT配方与87.54/75.10
作者表格，实际为LSTM decoder；历史wrapper归一化包含batch轴，原函数代数替身
验证batch耦合。见speechbrain-history-v1末节及historical-recipe-audit-v2。
不认证HF权重训练提交、不变更当前runtime、不新增下载/训练；SLURP权重复测仍缺资产。

配方引用论文2111.02735v1全文及Table2图核验发现PF-HuBERT为87.51/75.32，
与发布README87.54/75.10分开；论文GRU描述与发布LSTM不同，synthetic及SLU种子
数未明确。Frozen encoder行仍训练下游，不是零训练基座。上述比较边界归档于
speechbrain-history-v1及slurp-hubert-paper-source-audit-v1，不能把SF内容指标当
S_present类型指标，也不能跨官方TEST/clean/内部selection称RL收益。

NGA迁移方法核对见doc22第7节：固定源hash确认T3仅目标问题前缀干扰，非多Span监督；
原T2反转函数的三等级反例确认id/定义变化而gold未变化，不直接复用标签。
当前问句变换补unassessed全null及覆盖分母，13项变换/指标测试通过；不改变菜单、
人口、奖励、历史报告或语义准入。仍按任务定义→工程验收→训练推进。

损失分母接口差异已实证：旧hybrid回调除当批有效权重和，与新规约除真实请求数
不一致。现有CPU函数的权重0.5/1.5单请求反例使梯度均为0.25，而固定人口目标为
0.125/0.375；直接换分母参数也会被旧有效权重检查拒绝。doc36末节及
request-loss-reduction-audit-v1记录原函数/hash/边界。rl-v2补统一RL/CE/KL归约、
G样本平均、padding/DP累积要求及验收门槛；未修改历史actor、未接通标准算法。

FSC读出设计已补齐：transfer-v3引用独立native31/semantic31绑定、8面板及Direct/Q_input
位置，source Q_credit不增加目标臂。25项相关测试含11,904合法输出映射组合通过，
doc114归档；未读取新测试样本、未产生标签或派发推理。实际renderer/tokenizer及
音频上下文/后端验收仍缺失，不能把设计绑定视为迁移收益。
