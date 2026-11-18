# context-prob-v1：当前任务、工程与算法基线

本页是当前概率决策主线的执行入口，更新于2026-11-15（北京时间）。

最新普通观察为local1794694683.1253543：同一正式GRPO容器四rank完成step10/634，
running/非OOM，最近最慢101.075秒；下次普通观察>=1794695283.1253543。
见`context-arithmetic-formal-live-v3.json`。后文step4是更早观察。
新增TEST/FSC评测工具已在固定镜像无NPU环境完成110tests/30.63秒，容器
`0f96672fa411de4c` exit0/非OOM，源码、runner、终态和日志由独立收集器核验。
见`slurp-context-test-cpu-v1.json`。相关模型、来源与SSH为测试替身；不代表实际TEST前向或成绩。

执行顺序固定为：**任务与比较设计 → 工程及配置冻结 → 有限验收 → 正式训练**。
这里只描述当前有效约定；504行原入口已逐字归档到
[历史记录](context-prob-v1-baseline-before-20261115-consolidation.md)，不得用旧状态覆盖本页。
算法和step-wise详细论证见[研究执行决议](../../research/context-baseline-execution-decision-20261114.md)。

**最新执行状态（覆盖下方旧失败与验收等待记录）：**修复版本已通过真实DP4连续更新
及新进程恢复重放，四rank master/Adam/采样/执行/joint策略精确一致。
独立复审见`context-arithmetic-formal-acceptance-v2.json`；旧checkpoint频率证据仅复用
字节未变的保存/导出逻辑，不称为新策略轨迹证据。正式GRPO新版404文件已封存，
发布ID `493cc1bea2fafc93a18827707f11ee9cf5308ee7078990c1e52d9d425a422d49`。
driver599307启动容器`e564df04f946f853`，2026-11-14 21:56:41 UTC确认running/非OOM；
普通下次观察不早于1794694001.3869965。准确句柄由原设备租约下的launcher等待。
链路为634步→独立收集→新版完整训练审核→11点完整selection；尚无正式完整曲线或收益。
训练从原Stage1 step634重置optimizer开始，不使用工程两步后的权重。
见`context-arithmetic-formal-publication-v2.json`、`context-arithmetic-formal-start-v2.json`。
仅新增tools/reports版本化发布/运行/审核入口，原364文件身份未变；45局部CPU检查通过。

到期观察local1794694054.9590786确认同一GRPO容器仍running/非OOM，四rank完成step4；
最近最慢一步102.129秒、actor约23秒，仅单步live值，不当稳态吞吐或效果。
下一次普通观察不早于1794694654.9590786，见`context-arithmetic-formal-live-v2.json`。
FSC新版条件接续driver599824已确认存活：等待双曲线成功后才冻结GRPO初始/选中权重，
执行3793条FSC及独立评分、归档SpeechBrain比较，目标集不调参。
SLURP TEST双策略推理/打包/launcher/独立collector已准备，23+8+10项CPU检查通过，
部分模型/来源/SSH为替身；尚无实际TEST选中来源包、NPU推理或评分。复用12991条缓存，
两策略256条HF/vLLM一致性须先于全量推理，不能以CPU通过声称TEST成绩。

TEST评分与完整host编排现已接续，21项CPU检查通过，包含真实小型指标、标签映射及
运行审核失败时不读标签；编排的模型/来源/SSH为替身。预先封存的评分契约绑定12项
数据/参考身份，只有独立推理审核后才读取clean TEST标签。driver600506等待FSC
原owner成功后复用源选择pair执行，不使用FSC分数选择模型；尚无TEST实际模型执行。
见`slurp-context-test-scoring-preparation-v2.json`。全部运行与排队依赖身份复核无变。
已验收两步的耗时独立复算见`context-arithmetic-prefix-timing-v2.json`：第二步105.206秒，
参考策略首次计算33.998、其余rollout34.983、actor23.437秒。固定reference缓存按
policy及实际输入身份索引，后续相同输入可复用；尚无第二轮实测，不据此改当前配置。

## 1. 已经锁定什么，尚未发布什么

唯一冻结配置目录为`configs/releases/context-prob-v1/`：

- `experiment.json`：任务、数据、模型、损失、预算和评测。
- `algorithms.json`：GRPO主线、RLOO-clipped对照、PPO-critic设计预留。
- `LOCK.json`：源码、配置、固定镜像、模型、数据、日程及既有验收身份。

lock ID：`2ece49811d1e11eabc75f3e95e73ef532bda3fe9f16e9a399da7725ad0d95bfc`。
`python tools/jev/verify_context_baseline_lock.py`本次核验364项通过，`stage2_released=false`。
固定JSON中的状态是发布时快照，不原位修改其历史status；后续阶段由独立准入与终态证据说明。
**配置冻结、工程准入与效果证据分开记录。** Stage1及旧版本有限验收已通过；
原正式GRPO运行包首个optimizer更新前因行为log-prob一致性失败退出。
数值修复版本的真实四卡连续两步及新进程恢复现已独立审核通过，正式新版GRPO已运行。
完整634步及收益曲线尚未取得；不能称完整RL基座已稳定。原LOCK中的false为历史快照。

| 工作 | 已核验状态 | 依据 / 边界 |
| --- | --- | --- |
| 简单任务与全量输入 | 40513学习、9226选择、60候选已固定 | 已知SLURP缺陷隔离；TEST不参与优化和选模 |
| Stage1 | 完成634/634更新，exit0、非OOM，已独立收集 | ranking + binary NLL；不是RL |
| Stage1终点评测 | 完整9226选择集通过 | set-F1 73.8041%，top1 82.0724%，macro-F1 77.4315% |
| 训推一致性 | 该终点256条HF/vLLM概率差为0 | 不推广为所有输入、所有策略均已一致 |
| 连续RL与恢复 | 真实两步连续更新、独立进程恢复重放已收集 | 工程验收；不构成634步RL效果 |
| checkpoint间隔验收 | interval1/64两步对照通过 | 采样、逐步执行、末步master/Adam四rank精确相同 |
| 正式GRPO | 终态exit1、非OOM，完成0个optimizer更新 | 首批binary log-prob门槛失败；旧队列已退出 |
| 修复版本有限验收 | DP4连续两步及新进程恢复独立审核通过 | 四rank参数/Adam/采样/执行/joint策略精确一致 |
| 新版正式GRPO | 容器e564df04f946f853已确认running/非OOM | 634步与11点选择集链执行中，尚无完整曲线 |
| 新版RLOO-clipped | 404文件包封存，来源与源码同GRPO逐字一致 | driver599537等待原GRPO实验owner成功终态，再运行同预算对照和曲线比较 |
| 正式RL效果 | 尚无完整曲线 | GRPO/RLOO各634步、11点评测仍待 |

终点评测证据：`docs/engineering/context-stage1-selection-step0634-v2.json`。
其NLL=0.0187434286、Brier=0.0046499592；不得把该成绩称作TEST成绩或RL增益。
原收集器的汇总浮点末位问题由独立v3收集器修复并复用原产物；仅NLL/Brier允许8 binary64 ULP，
身份、概率和计数仍精确核对。详见`context-metric-audit-repair-v1.json`。

初始化完整9226选择集亦已收集，原容器exit0/非OOM。初始set-F1=3.3191%、
top1=1.7993%；初末9254项产物各自hash复核、全部音频顺序/输入张量身份/位置相同。
初始头平均返回59.25个候选，Stage1终点空集1703条、单选7328条、多选195条。
这是新头Stage1适配前后比较，不是原生ASR零样本成绩或RL收益；详见
`context-stage1-initial-endpoint-comparison-v1.json`。正式RL的step0仍是Stage1 step634。
checkpoint容器`3d877a4c4c75`已exit0/非OOM，独立收集与间隔比较通过，见
`context-checkpoint-comparison-npu-v3.json`。原sequence v3实际失败于host子进程未设置
PYTHONPATH，`ModuleNotFoundError: multimodal_jev`，尚未派发任何正式模型。
本次仅补启动环境，复用同一507pins配置和已有产物，在独立sequence v4接续；来源绑定及
正式GRPO发布已通过，容器`19f100b20fd744d9`随后于2026-11-14 21:16:24 UTC
终态exit1/非OOM，四rank均在update0的optimizer之前停止。
发布ID：`2689659cde35e5233b77a66d5c51ff50e9a843e28787c196eaf523397b4cdb89`。
启动历史保留于`context-formal-launch-environment-repair-v1.json`；终态与失败证据见
`context-formal-behavior-parity-failure-v1.json`。不改旧现场、不重跑Stage1。

实际失败rank的binary log-prob差4.0531158e-6，超过冻结2e-6；概率误差8.4971e-8
在2e-7以内，阈值翻转0。源码中actor使用logsigmoid(logit)，rollout从实际FP32
sigmoid概率反算log/log1p。原128音频/原策略单卡NPU重放已exit0，并独立复算：
全部7680候选的实际actor概率与归档vLLM概率精确相同；原路径16块中3块失败，
最大log差4.0531158e-6。统一为NPU FP32 sigmoid→CPU可微log/log1p→原设备的
诊断候选log差为0，小logit梯度有限且与稳定解析路径最大差1.78814e-7。
见`context-probability-arithmetic-npu-v1.json`。证据支持本次是数值表示路径不统一，
并非模型前向分歧或OOM；没有镜像损坏证据。阈值未变；该修复的主干反向与
FSDP连续更新及独立恢复现已通过验收。生产修复已版本化，原包保留。

修复候选已封装为`context-bernoulli-probability-cpu-log-v2`：副本仅替换actor算术
import并给checkpoint增加数值协议身份，原固定配置和364文件不改。48项局部CPU检查
通过；实际DP4前两批候选`context-arithmetic-grpo-prefix-v2`容器
`d3f7168491a03764`已exit0/非OOM，326项产物独立收集并审核通过：两批各128条，
四rank行为log差为0，主干/LoRA/head反向、optimizer、更新后策略激活及BF16导出均核验。
证据：`context-arithmetic-grpo-prefix-v2.json`。两步总耗时分别110.590/105.206秒，
其中rank0 actor为25.997/23.437秒；仅为有限验收，不作为稳态吞吐或RL收益。
新进程恢复容器`813de6ba21b28ffa`已exit0/非OOM，从step1重放step2，178项产物收齐，
独立审核及完整master/Adam/采样/执行/joint权重精确比较通过；driver598135正常结束。
证据：`context-arithmetic-replay-comparison-v2.json`、`context-arithmetic-formal-acceptance-v2.json`。
新数值身份已贯通版本化正式运行、保存和审核入口；两算法共同使用该版本，从原Stage1终点开始。

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
[runtime-storage-policy.md](../runtime-storage-policy.md)。普通远程观察至少间隔600秒。
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
两者都是概率动作适配；不是原论文自回归token配方逐项复现，RLOO-clipped也不是未改动的REINFORCE。
本轮检验混合RL方案相对Stage2 step0的持续改善，不新增RL-vs-SFT比赛，也不声称已隔离纯RL因果贡献。

## 5. Step-wise先设计信用与监督，不先改标准算法

Q1/Span1、Q2/Span2、Final/Span3是输入预设、一次前向的并行判决；问题编号不是环境时间。
当前Final不会读取同层刚输出的其他概率，所以不沿Q编号套temporal GAE或token reward-to-go。
如果未来要求前置prob参与Final，必须先定义内部依赖或harness聚合，再冻结新工程；不能口头假定已有。

复杂任务仅在Q的可迁移语义、合法动作、过程真值及缺标mask均固定后开展三组比较：

1. Direct：直接Final判决。
2. Q-input：增加统一语义问题，只使用Final reward。
3. Q-credit：与Q-input相同模板、动作、过程辅助监督及预算，仅增加可靠过程reward。

Q不得依据当前query答案或gold选题；缺标mask，不补false。Final标签的确定性投影明确称属性辅助监督。
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

1. 初始化与终点完整选择集均已独立收集；不重跑已完成Stage1。
2. checkpoint interval1/64逐步策略与末步master/Adam核验已完成，不重跑。
3. 共同Stage1来源及已有连续/恢复/保存证据已绑定；正式首批失败后，仅处理log-prob
   数值路径阻断点。先原批次重放，再版本化修复、验证受影响前后向/分布式更新及恢复。
   已通过且未受影响的缓存、Stage1和评测证据复用，不无条件重做全部验收。
4. 新数值版本准入后，两算法同用新版本、同Stage1起点；GRPO完整634步，再运行
   同预算RLOO-clipped。完整9226选择集在
   `0,64,128,192,256,320,384,448,512,576,634`共11点评测，报告全曲线、退化、概率质量和成本。
5. 简单任务闭环后开展冻结权重的FSC配置迁移，以及有可靠监督的复杂Q实验。

来源绑定、发布已在真实产物上执行通过；sequence v4与comparison v6、FSC v2、
reference v3已因正式训练失败退出，不能继续描述为等待或运行中。修复验收前不重排正式链。
启动环境固定含repo/src、repo/.local/verification-python、repo/tools/jev及repo；
新鲜host子进程实际导入和来源打包已验证，不能用仅mock的编排测试替代这些检查。
运行容器仍使用固定镜像与自身显式环境，host修复未改变数值训练代码。
不再增加micro32吞吐试验、PPO、复杂Q、FSC或新增学术模型下载作为简单主线前置条件。
仅修复阻断既有闭环的实际缺陷；运行/排队包禁止原位修改，失败保留准确终态和产物。
任务、候选、损失、reward、batch、精度或数据日程变化须新实验身份，不拼接曲线。
编排/收集修复用独立版本和受影响范围验证；可重收集已有合格产物时不重新训练。

发布收口采用以下固定边界，避免把一次验收变成不断扩展的工程项目：

| 阶段 | 必须形成的产物 | 本轮状态 |
| --- | --- | --- |
| 任务冻结 | 输入/候选/合法动作、监督边界、奖励、指标、划分、完整预算 | 简单任务已冻结；复杂Q尚未发布 |
| 工程封存 | 镜像digest、依赖/源码hash、模型与processor、数值协议、明确入口 | 原版已封存；已定位数值补丁另立版本 |
| 有限验收 | 固定原批次概率/log-prob、真实DP4更新、策略刷新、保存及独立恢复 | 原证据复用；数值补丁只重验受影响链路 |
| 正式实验 | 同一版本GRPO/RLOO，634步与预定11点评测，独立收集 | 受数值补丁验收阻断；尚无RL收益 |

正式run期间不调reward、模板、精度、microbatch或评测口径；若必须变更则结束旧身份，
保留完整失败/终态，新身份从明确起点开始。固定配置不是效果承诺；训推一致性、
工程稳定性和学术提升分别验收。新增吞吐优化不插入上述有限准入链。

另有同Omni clean TEST人口的12991条Jev输入已经准备、独立重建，见
`slurp-context-clean-test-inputs-v1.json`；完整CPU processor及新进程缓存完整性检查均已
exit0，12991条、最大3081 tokens，见`slurp-clean-test-processor-cpu-v4.json`。
这是原始processor执行及独立缓存核验，不是独立重算processor、模型前向或TEST成绩。
它与学习/选择集无ID或压缩音频文件hash重叠，不代表完成语义去重。该输入准备不阻塞当前RL。

## 方法依据

DeepSeekMath（2402.03300v3）§4.1.2/4.1.3；Back to Basics / RLOO（2402.14740v2）§2.3；
Proximal Policy Optimization Algorithms（1707.06347）。2026-11-15已重取原文核对，
本地来源及hash见`.local/decision-objective-research/baseline-method-sources-20261115/receipt.json`。
在线原文用于核对方法；执行数值以本地固定源码hash为准，不随上游latest升级。
原论文效果不提供本项目效果证据；本项目结论须来自上述冻结预算的实际曲线。
