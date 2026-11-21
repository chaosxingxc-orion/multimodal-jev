# Stage2 末点错误诊断与下一轮简单任务实验

2026-11-19。用户决定本轮效率优化收口，下一优先级为末点bad cases和新一轮训练。
完整评测任务组已测得同容器边界2.065倍；不声称10倍，也不再以10倍作为新实验前置。
本说明的分数来自完整9226条selection（官方TRAIN来源的独立选择集），不是TEST。
分析目标固定末点634，未将其称为完整曲线上的最佳检查点。

## 已核实的结果

训练独立审核已完成：634更新、81026真实样本暴露、DP2；reference零在线补算。
证据 `operations/context-stage2-formal-v0.0.93/finalization/audit.json`。
错误诊断使用固定脚本 `tools.reports.decision_error_diagnosis`；核验发布清单、完整人口、
预测身份和原评分，再核验634份TRAIN轨迹与已审核传输清单。每轮真实TRAIN恰为40513条，
尾批padding按ID排除重复。报告位于 `/srv/multimodal-jev-xingchao/operations/stage2-endpoint-error-diagnosis-v2/`。
逐样本结果只保留在该外部报告，不把selection样本或ID纳入下一轮训练配置。

| 指标 | Stage1终点 / Stage2初始 | Stage2末点634 |
| --- | ---: | ---: |
| Top1 | 82.224% | 83.221% |
| Top1 macro-F1 | 76.703% | 78.736% |
| 集合F1（固定0.5阈值） | 61.439% | 67.383% |
| 集合完全正确 | 28.008% | 31.476% |
| 平均返回候选数 | 1.966 | 2.175 |
| 真值召回率 | 86.939% | 94.385% |
| 候选平均NLL，越低越好 | 0.055010 | 0.079583 |
| 候选平均Brier，越低越好 | 0.015307 | 0.017856 |
| 包含概率>0.99错误候选的录音 | 155 | 2719 |
| 真值概率<0.01的录音 | 110 | 183 |

RL阶段提高了命中与排序，但概率质量退化。Top1的0.997个百分点提升与集合F1的5.944个百分点
提升是本轮混合目标的观测，不能全部归因于纯RL，尚未做跨seed因果结论。

末点9226条互斥分组：完全正确2904；真值加多余候选5804；未命中真值但返回其他候选518；
空集合0。集合错误6322条中，5804条（91.81%）是多选。4774条Top1已经正确但集合错误。
初始到末点：1250条“真值加误选”变为完全正确，同时956条原本完全正确退化成多选。
不能只展示前者。错误候选总数10117→11362，平均每条负候选NLL之和2.943→4.537；
正候选NLL反而0.358→0.238，说明NLL恶化主要来自负候选侧。

排序错误的频繁类别对包括general_quirky→qa_factoid（34）、calendar_query→calendar_set（32）、
general_quirky→qa_definition（30）、audio_volume_other→audio_volume_up（22）。
play_podcasts→transport_query（21）不宜直接解释成语义接近：需要TRAIN同类失败中的音频、
候选定义和原始可信标注证据后才能区分声学、定义、数据或模型因素。没有听音/独立标注复核，
本报告不虚构“口音/静音/标签错了”的根因，也不为这些类别手写评测专属reward。

真实输出例子（固定ID排序取例，不是人工挑选“好解释”的样本）：lists_query真值概率0.999685，
错误calendar_query仍为0.985936，Top1对但harness返回两个候选；play_podcasts真值0.999704，
错误play_audiobook为0.988313，也是概率排斥不足。另一条alarm_remove真值仅0.000667，
错误calendar_remove为0.999869，属于排序与存在性都错。这些例子仅说明输出机制，
不凭概率解释原始音频内容，也不将selection录音加入训练。可追溯ID见外部逐样本报告。

## TRAIN上的独立支持

两轮均覆盖全部40513录音，但behavior随更新变化，以下不是固定末点TRAIN成绩。

| 轨迹诊断 | 固定Stage1 reference | 第一轮behavior | 第二轮behavior |
| --- | ---: | ---: | ---: |
| 真值加误选 | 24451 | 23798 | 20039 |
| 完全正确 | 11779 | 14135 | 18495 |
| 含>0.99的错误候选 | 518 | 6526 | 6023 |
| 真值<0.01 | 212 | 653 | 460 |
| NLL | 0.053289 | 0.059894 | 0.048757 |

TRAIN第二轮总体改善，selection仍明显过度自信，因此不能据TRAIN最终窗口断言泛化校准改善。
这些TRAIN信号足以支持通用的概率纠错、集合效用与难负例排序方案；不需要从selection选出难例。
全量人口与每轮曝光保持，第一版不额外重采样、不按评测混淆对调整类别权重。

## 工程与目标机制的根因

1. **相对排序不约束绝对概率。** Stage1使用InfoNCE；同时平移所有logit不改变排序loss。
   正确候选第一名并不意味着其余候选低于0.5。独立sigmoid必须另有概率质量目标。
2. **当前二路奖励没有完整集合效用。** 节点存在性reward为bit正确与否；正负权重各0.5，
   59个负候选分摊负侧权重。排序reward也平均分到59对。稀少的高分干扰候选容易被稀释。
   这不是节点权重实现错误，而是现行目标与harness完整集合效用的差距。
3. **极端误判会使采样组无差异。** G=16时，错误p=0.99的bit全部采为1概率为0.99^16≈85.15%；
   p=0.999时≈98.41%。这些组reward恒定，中心化优势为0。乘大reward不能恢复零信号。
   Brier对logit的梯度也含p(1-p)，单独用Brier不适合解救饱和误判；稳定logit NLL更直接。
4. **原NLL与KL归一化不一致。** 源码为0.3×候选平均NLL，而KL为0.01×候选KL之和。
   60候选时每bit NLL系数只有0.005，KL为0.01。Stage1 reference尚未校准，强锁定其绝对概率
   可能阻碍校准。只能称机制风险，尚未用消融证明其因果贡献。
5. **未知语义/声学根因不能靠reward编造。** 仅对可信完整exclusive-event监督训练；
   受hold标签继续禁止。单标签数据中“错误但语义接近”不给未经监督的部分正确奖励。

## 下一版目标：保留双信用，增加集合效用

可编辑文件 `extensions/stage2_task_objective.py`；共享参数在
`configs/experiments/simple-task-next-round-v2.json`。所有系数为预登记起点，尚未证明最优。
新包按文件SHA和options绑定，冻结副本不可编辑。固定镜像CPU已有原生verl接口验证，NPU待验收。

采样不增加声学前向：同一音频的一次logit向量，保留排序/存在性采样；存在性各候选使用独立
随机流，其第g列构成真实独立Bernoulli完整集合A_g。任务reward：

`R_set(A,Y) = 0.5 * 1[A=Y] + 0.5 * F1(A,Y)`。

只有正确意图得1；正确+1误选得1/3；正确+2误选得1/4；纯错误/空集得0。
组内标准差下限0.25，clip沿用0.2。任务ratio使用完整集合的logprob求和，不能将reward
广播到节点后冒称joint GRPO。排序和存在性仍使用各自原ratio。

`L = .25 L_rank_GRPO + .25 L_exist_GRPO + .50 L_set_GRPO`
`    + .10 softplus(max(z_negative)-z_positive)`
`    + .05 SUM_c BCEWithLogits(z_c,y_c) + .01 MEAN_c KL(Bern(p_c)||Bern(p_ref,c))`。

NLL仍是独立Bernoulli正确评分辅助项，不把模型改为封闭分类softmax；没有候选间归一化。
均匀NLL不使用按真值类别重加权来伪装校准；SUM明确记名，其系数相对旧60候选实现提高10倍。
KL改MEAN，系数保持显式；与原值不可按同一数字误称同一强度。最大负例pair项仅用TRAIN可信监督。
在60候选源任务上，当前候选的KL强度实际降为旧版1/60，这是较大改动，不能假定一定更好；
有限验收及正式曲线必须记录相对reference的KL和饱和误判，遇到数值失败立即停止并保留证据。
下一次权重调整也必须新建实验身份，不能根据selection的坏样本在线修改本轮系数。
GRPO优化决策效用不自动保证概率校准，NLL/Brier和饱和错误数量始终单列。
首版不增加熵奖励、不修改阈值、不自动丢弃难样本，避免同时混入更多机制。

额外观测：三路loss、task组零方差比例、采样exact/partial数、NLL总和、最难pair、KL均值。
NPU有限验收还须确认joint ratio、有效梯度、padding、DP归一化与新进程恢复；CPU单测不替代它们。

## 上下文和运行安排

新布局：`固定规约 + 完整候选定义/问句 → 音频 → 固定顺序的decision_id/marker`。
所有marker依然看到完整音频和全部定义；不生成CoT、不串联转写模型、不按gold改问句。
输入缓存身份包含布局。新Stage1从原已审核初始策略开始，四卡InfoNCE、全量两轮318更新、warmup16。
不能把旧末点直接加载新模板当同一实验，也不能复用旧reference概率。

新Stage2使用新Stage1终点和全TRAIN reference；两训练卡+两个单卡推理实例，BF16、q/v LoRA r4/alpha8，
显式复用Stage1 adapter，global batch=64×DP=128，micro8。共634更新，正式训练内统计效率。
服务采用已测试的batch64、KV24GiB、token预算196608；输入驻留、长度分桶、每rollout完整交一个实例。
prefix cache与head历史sidecar在同一冻结策略内复用，LoRA更新必须失效；冷/热一致不代表与actor一致。
现有native缓存候选与冻结对齐runtime仍有数值差异，不能把它作为已验收服务直接投产。

固定四卡评测、一次加载语料、多checkpoint常驻切换。完整selection曲线按预登记set-F1/NLL/最早步选模；
TEST及FSC/MINDS只用冻结策略。新旧整套方案的差值包含布局与目标变化，不能单独归因reward。
若后续要声明某个loss贡献，使用同一新布局的预登记消融，仍不做RL对SFT比较。

## 当前交付与尚未完成的工程

已完成：完整末点诊断、两轮全TRAIN轨迹核验、可编辑目标实现、CPU语义/梯度测试、下一轮参数设计。
固定镜像CPU的25项验收证据在 `operations/simple-task-objective-native-cpu-v0.0.108/report.json`，
对应冻结副本；后续源码修订须重新绑定，不能沿用旧hash宣称通过。
扩展回归封包v109/v110分别遗漏历史测试配置/fixture依赖，保留失败证据在同名operations目录；
已补齐为独立v111，不把封包错误表述为NPU训练失败。
v111已独立收集审核：固定镜像88 tests、0 failures/errors/skipped，包含真实pinned verl目标
和新布局相关回归。最新目标SHA256为`0dcd7703b6a7a5c4b579c56cecc79954136426dda6b4eaacc636f3cc3a9b9c3c`；
证据 `operations/simple-task-objective-native-cpu-v0.0.111/report.json`。这是CPU验收，非模型NPU验收。

尚未启动新训练。实际发现的接入缺口必须先修复：

- 数据预处理/Stage1/reference/评测编码器已补充`prompt_layout`传递，本地相关回归87项通过；
  发布器的完整配置传播、新布局缓存与真实processor输入证据仍需完成。
- 现有正式Stage2封包与审核仍限制默认objective字节及空options；必须改为核验新plugin身份和实际计算，
  不能只删掉门禁后让旧公式审核新loss，也不能伪造已完成的有限验收。
- 新布局的actor/rollout与前缀hidden-state恢复需要真实NPU验收。

执行顺序为新布局输入准备和固定镜像CPU并行→两更新/恢复/真实尾批→正式Stage1→新reference→
新Stage2有限验收→正式Stage2。必要验收通过后按既有授权直接推进，无需再次请求启动许可。
