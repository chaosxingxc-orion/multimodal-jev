# 原生属性概率判决：全量候选数据与输入容量

本候选用于检验“可核验的属性分解是否改善Final”，补充真正语义Q研究。
所有属性真值来自已清理Final标签的确定性投影，不是独立语义存在性或过程推理标注。
当前正式GRPO/RLOO实验、输入、奖励、预算与队列没有更改。

## 已制作和独立复核

位置：`.local/decision-objective-research/slurp-native-attribute-prob-candidate-v1`。
输入文件仅含音频、音频hash、记录ID及全局catalog引用；targets单独存储。
全40,513条TRAIN、9,181来源组，统一18个scenario组、46个action组、60个Final组，
共5,023,612个判决标签；每个样本每族恰有一个正例，共121,539个正例。
独立脚本不调用生成器的targets函数，从原始scenario/action/intent重新计算全量标签、
组成员关系、音频人口及输入hash，全部通过。结构歧义13,679条，其他26,834条。
没有读取selection/TEST标签用于定义或选样，也没有新增模型推理。

配置入口为`configs/tasks/slurp/native-attribute-prob-candidate-v1.json`，生成器为
`tools/reports/build_context_attribute_candidate.py`。审核收据：
[全量标签审核](slurp-native-attribute-prob-candidate-audit-v1.json)。

## 问句、对照和奖励边界

统一问句为“按照给定事件规约，音频的目标事件是否属于这个定义的组？”
组由完整事件目录和明确成员ID确定，不以当前gold决定问什么。
同一事件可以同时属于一个scenario组、一个action组和一个Final单元素组。
这些prob分别读出，不强制整个124维向量总和为1，不通过属性AND/乘积修正Final。

Direct只读出Final60；Q-input/Q-credit同一次前向读出完整124个概率，两者请求完全相同。
三个实验共享完整事件目录及Final定义，保持同样Final辅助loss，均不增加过程辅助loss。
Direct与当前正式Final60的输入表示不同，需要新的匹配Direct对照，不能直接挪用旧成绩。

Q-credit候选reward为Final集合F1加lambda乘两属性族集合F1的均值；lambda尚未冻结。
不以大量负位的accuracy代替集合奖励，不因action候选更多而让其取得更高族权重。
GRPO标准化会放大小过程奖励的相对作用；必须结合Final收益及优势尺度评估。
此控制检验标签分解贡献，不宣称隐式思维链、因果过程信用或跨数据集语义已验证。

## 实际固定镜像CPU processor结果

同52ba94e1镜像，无NPU和模型权重，实际加载已冻结processor，使用TRAIN音频
样本数最短、中位、最长三个锚点。5个编译单测通过，无skip；6个请求全部编码完成，
Q-credit与Q-input请求逐字段相同。1080包文件及实际终态、日志、result独立收集。
容器3862c083ed619612正常exit0/非OOM。

| 音频锚点 | 音频tokens | Direct tokens / 60读出 | Q-input tokens / 124读出 |
| --- | ---: | ---: | ---: |
| 最短，8191采样点 | 7 | 4546 | 8086 |
| 中位，41979采样点 | 34 | 4573 | 8113 |
| 最长，373760采样点 | 304 | 4843 | 8383 |

当前4096上下文无法容纳这些候选，最长Q甚至超过8192。两种实验臂非音频长度分别
固定4539和8079；三个锚点的长度差与音频token差相符。主要成本来自重复表达问句、
组定义和读出，而不是音频长度。当前不扩大上下文、不截断规则或音频，也不开始训练。
下一步先做保留目录/组成员/问题语义的紧凑表示，再明确上下文与实际资源预算。

这只是三个真实音频锚点的processor测量，不是全部人口、数据增强、模型forward或
训练显存验收。证据：[固定镜像CPU收据](context-attribute-processor-fixed-image-cpu-v1.json)，
原始result位于`.local/decision-objective-research/context-attribute-processor-cpu-v1-collected/run/result.json`。

## 紧凑表示：实测落入4096，尚未准入模型

独立编译器`tools/reports/context_attribute_compact_candidate.py`将共同问句仅存一次，
事件目录改为[id, definition]表，组成员使用显式零基事件索引，读出按组索引对齐。
没有删减事件、判决、成员、定义或规则。pack/unpack逐字段还原原请求；全真实目录
60事件/124组另经独立字段重建检查，Q-input/Q-credit表示完全相同。
数字索引只是定义引用，不是类别专属head参数或由gold选择问题。

固定同一52ba94e1镜像CPU容器491f123b5fd28636正常exit0/非OOM，12tests全部通过无skip。
1082项包文件及终态/result独立收集；与前次实际audio、catalog、processor字节hash一致。

| 音频锚点 | 紧凑Direct | 紧凑Q-input | Q-input减少 |
| --- | ---: | ---: | ---: |
| 最短 | 1981 | 3018 | 5068 tokens / 62.68% |
| 中位 | 2008 | 3045 | 5068 tokens / 62.47% |
| 最长 | 2278 | 3315 | 5068 tokens / 60.46% |

全部60/124读出位置均保留；各锚点的非音频长度、读出相对音频偏移一致。
真实配对审核见[长度对照](context-attribute-compact-comparison-v1.json)，
[字段复原](context-attribute-compact-field-audit-v1.json)及
[固定镜像CPU](context-attribute-compact-processor-fixed-image-cpu-v1.json)。

最新设计为`configs/tasks/slurp/native-attribute-prob-candidate-v2.json`，复用相同候选数据
和全部标签，不重新标注。三个锚点落入4096不等于全人口processor或训练显存已验；
字段可逆也不等于模型对数字索引和原文字定义表现相同。仍需全人口输入核验、
选择集协议、过程reward权重/预算及新实验的有限NPU验收。当前不改正式配置，不派发训练。

## 全量输入审核已启动，结果待收集

固定镜像52ba94e1、2 CPU/8GB、无NPU的独立容器
`edf4509f7401eb3ea11fb05f86db2363a0ae81fd6be01fcd4e2ac72ee79804ab`
已启动全40,513条TRAIN音频审核，计划实际执行81,026个Direct/Q-input processor请求。
审核音频字节与波形身份、两臂音频特征一致性、60/124读出及4096长度约束，
仅落盘记录与直方图，不新建大规模特征缓存。Q-credit复用相同Q-input请求。

驱动704649保持准确容器句柄，终态后接收集与独立全人口复核；本轮本地状态为
`wait_exact_cpu_container`，1073项依赖身份校验通过。尚无全量成功或独立审核结论，
不得将三个锚点结果替代全人口证据。独立复核会校验完整记录与长度/位置，
不重新执行全部音频processor，因此特征hash仍属于执行收据。
运行包和审核脚本保持不变；该任务不占NPU、不阻断当前简单RL实验。

## 选择集与首轮比较协议

选择集已独立制作到`.local/decision-objective-research/slurp-native-attribute-selection-candidate-v1`，
9,226录音、2,146来源组、124判决，共1,144,024标签。目录与TRAIN逐字节相同，
没有按选择集标签改问题。独立从原scenario/action/intent重算全部标签通过；
录音ID、来源组、编码音频SHA三种身份与TRAIN均无交集。10项本地检查通过。
证据见[独立标签复核](context-attribute-selection-independent-v1.json)。这不是选择集
processor或模型评测结果；尚无复杂任务成绩，未读取TEST。

首轮比较设计固定为`configs/experiments/context-native-attribute-comparison-design-v1.json`。
它补充此前任务候选中未定的实验参数，不修改旧候选、当前运行或排队依赖：

- 三臂均从同一个简单Stage1终点开始，不各自增加预训练；固定reference使用同一权重，
  对各臂实际请求分别前向。必须各自报告step0，不能假设相同权重在不同输入上起点指标相同。
- 第一轮lambda=0.25，过程奖励为两属性族F1均值，最大额外原始奖励0.25；
  这是事先选定的预算，不是最优性或GRPO梯度上界证明。本轮不做lambda搜索。
- Q-input与Q-credit均采样完整124-bit动作，使用同样联合ratio、KL支持集、Final辅助loss；
  两者只改变reward。Direct为60-bit。Direct与Q-input同时改变长度、动作维度及KL支持，
  因而只能支持整体配置对照，不能声称单独隔离了问句文本的因果收益。
- 每臂同样全量两轮634更新、global128、G16、11个完整选择点评测。主对照先报告
  step634的Q-credit减Q-input，再报告各自按Final指标选出的checkpoint及其step0变化。
  同时报告时间/token成本，不把等曝光称等算力。Final set-F1@0.5最高、NLL最低、最早步选模；
  过程指标不参与选模，不使用属性乘积、AND或其他后处理修复Final。
- GRPO为主、RLOO-clipped为匹配确认；两者都不能沿并行问题编号做temporal GAE。
  选择输出可按source_group做配对聚类bootstrap，但单种子不能证明训练随机性的稳健收益。

标签全量完整，此实验拒绝部分任务族标签，不将缺失补false。后续语义缺标任务需单独
冻结mask/reward规则。新工程还需124节点动作/损失/KL绑定和真实NPU有限验收；
以上只是前瞻设计，dispatch仍为false，不抢占简单基线设备或追加其发布条件。

## 奖励候选已验证，尚未接入actor

独立实现`tools/reports/context_attribute_reward_candidate.py`复用既有集合F1和排序/NLL
函数，按catalog切分scenario/action/Final三族。每族必须完整划分事件目录，Final为单元素组；
真值须与Final指定事件的组成员关系一致，拒绝缺标和彼此冲突的属性标签。
Q-input/Q-credit均接收完整124-bit动作，Direct接收Final60；不做属性到Final的修复。
Final辅助函数只切Final logits，因此对过程读出没有直接辅助梯度；共享主干仍可间接变化。

本地torch2.10.0+cpu的10项检查通过，包括三事件小目录全部128动作与Python集合独立比对、
节点顺序置换、非法/缺失/冲突标签拒绝、Final辅助梯度与Direct一致。
另在真实TRAIN40513与selection9226的全部6,167,636标签上执行四种合成动作控制：

| 控制动作 | Q-input/Direct Final奖励 | Q-credit总奖励 |
| --- | ---: | ---: |
| 三族全部正确 | 1 | 1.25 |
| 全部不选 | 0 | 0 |
| 全部选中 | 2/61 | 2/61 + (2/19 + 2/47)/8，约0.0513 |
| 仅Final正确，过程全不选 | 1 | 1 |

全部记录均通过；与对应FP32参考表达式最大误差0。Q-input与Direct使用相同Final动作时
奖励逐元素相同，Q-credit减Q-input恰为两属性族奖励均值乘0.25。
证据见[全量奖励核验](context-attribute-reward-population-cpu-v1.json)。
这些动作是人为构造的控制，不是模型采样；没有新增音频前向、NPU更新或学习效果。
尚未完成固定镜像检查、124节点actor绑定、联合策略/KL及真实更新恢复验收。
Q-input的过程动作虽不影响奖励，仍进入联合ratio，有限采样可能增加梯度方差；
此影响由Q-input/Q-credit共享，不应误报为时序信用或多步推理收益。

### 固定镜像复核完成

同正式52ba94e1镜像的独立CPU容器
`960831eddd213ade02e61ea248bcb7407c97823e7f9a6b3375eef897c35bd493`
已正常exit0/非OOM，无NPU设备、无网络。374项包身份通过，36项测试全部通过、无skip；
包括选择集隔离检查、目录投影、集合奖励、Final辅助梯度及既有辅助loss边界。
随后在同容器重跑全部TRAIN/selection四动作控制，49,739记录/6,167,636标签通过；
人口计数与全部控制结果同本地报告逐字段相同，最大奖励误差0。
测试加全量控制执行约66.11秒，容器总寿命约94.77秒；不是模型吞吐指标。

实际命令、镜像、无设备/无网络范围、退出状态和收集文件hash均经独立核验，见
[固定镜像奖励检查](context-attribute-reward-tools-fixed-image-cpu-v1.json)。
原始产物在`.local/decision-objective-research/context-attribute-reward-tools-cpu-v1-collected`。
本结果补齐固定镜像CPU检查；尚缺124节点actor/联合策略/KL绑定及真实NPU验收，
没有更改运行配置、没有训练收益或复杂任务派发。

### 联合损失候选：真实verl的CPU验证

`tools/reports/context_attribute_objective_candidate.py`新增独立候选，不注册或修改现有actor。
prepare_groups按完整G16先计算族奖励与GRPO/RLOO优势；loss采用已准入的
`context-bernoulli-probability-cpu-log-v2`概率路径和冻结verl clipped surrogate。
联合log概率与KL覆盖该臂全部60/124节点，旧categorical监督显式置零，
唯一辅助监督为Final的pairwise ranking及binary NLL。全局真实query分母及DP缩放沿用既有约定。

固定52ba94e1镜像CPU容器
`dba19e44902fe0d401c62a8801a4eedc4f127778d9300d5d105285257540ec03`
已exit0/非OOM；377包身份与产物独立收集，49tests全过，无skip。
新增13项检查直接调用真实、源码hash绑定的verl，无estimator/surrogate替身：

- 实际18/46/60目录，GRPO/RLOO × Direct/Q-input/Q-credit六种组合，
  独立展开联合ratio、dual clip、Bernoulli KL及Final辅助项，loss和logit梯度在声明容差内一致。
- 分三份计算并模拟rank平均归约，与完整批次梯度一致；尾部零权重补齐样本梯度为零。
  这是CPU代数归约检查，不是实际分布式collective。
- 相同优势输入时Q-input/Q-credit的loss与梯度逐元素相同，保证两臂差异只从reward进入。
- 奖励全同组的优势为零，KL/Final辅助仍可更新；缺标、附着梯度的behavior、
  错误动作支持、G8及缺reference均拒绝。

证据：[固定镜像联合损失检查](context-attribute-objective-tools-fixed-image-cpu-v1.json)。
使用的是合成logits/actions和真实任务目录，没有音频/model前向、LoRA反向或NPU更新。
运行时输入/目录/策略/reference身份、优势搬运及全局分母认证仍由后续actor候选负责，
本函数不是完整训练准入。下一步是独立actor绑定与真实更新/恢复验收；冻结运行包保持。


### Actor身份与传输验证完成，四rank更新验证进行中

独立`tools/reports/context_attribute_actor_candidate.py`从hash固定的原生actor构造候选，
不修改原模块或注册名称；三个分支分别绑定前瞻设计、catalog、节点顺序、G16和固定
Stage1 reference。每个分支有独立loss注册名及ContextVar，防止同进程串用策略。

固定52ba94e1镜像CPU容器9bb9bf3a9d1a50b0已exit0/非OOM，379项包身份独立收集，
10项检查通过、无skip。实际verl DataProto传输和单rank Gloo准入覆盖三分支reward/
advantage/old_joint一致性，以及错误分支、节点顺序、reference、G8、catalog及篡改
reward/advantage/分母/response mask的拒绝。输入概率/动作是合成值，没有模型更新。
证据：[actor传输与准入](context-attribute-actor-tools-fixed-image-cpu-v1.json)。

随后启动独立四rank CPU更新探针，容器df43bf37a89f8b65d7e46e608913792b4d64ea78f9384f1a4bc517766994fea1，
同镜像、4CPU/16GB、无NPU/网络，382项包身份封存。覆盖三个分支、双算法、micro1/2、
三次连续更新，比较实际verl/DDP与独立全批次loss的梯度/参数/Adam，并测试保存加载后
再次更新及跨rank损坏拒绝。使用合成音频主干与真实浅Transformer head；即使通过也
仅证明该CPU链路，同进程保存加载不等于新进程恢复，且不是Qwen/NPU/训练收益证据。
当前持准确docker wait句柄等待终态，没有复杂任务正式训练派发。

全TRAIN实际processor普通观察local1794707194.646185仍running/nonOOM，完成3072/40513；
下一普通观察不得早于1794707794.646185，原driver704649继续持准确句柄并负责收集审核。
正式训练1078项及processor1073项依赖均再次校验无变化。


### 四rank实际更新终态：通过

上述df43bf37a89f8b65容器已exit0/非OOM，准确wait返回0；382项包身份及实际命令、
镜像、无NPU/网络范围、terminal、四rank结果和日志均独立收集。
三个分支×GRPO/RLOO×micro1/2×三次更新，每rank36例均通过。
每批8行含7条真实请求与1条零权重padding，实际Gloo/DDP归约和真实verl更新循环执行；
独立参考自行展开三族集合F1、优势、联合clip、KL和Final排序/NLL，不调用候选reward/loss。
四rank相对于全批次参考的最大梯度差1.0728836059570312e-6、参数差1.8649029698281083e-7，
Adam一阶矩差4.470348358154297e-8、二阶矩差3.2014213502407074e-10，
全部满足预先沿用的atol2e-6/rtol2e-4。各rank真实behavior对齐检查通过；分母、mask、
padding、跨rank重复ID、contract、节点顺序、reward、advantage损坏均被集体拒绝。
第二次更新后保存模型/Adam，扰动参数再加载，第三次更新继续与未中断参考一致。
这属于同进程保存加载后更新，不扩大为新进程恢复证明。

证据：[四rank实际actor更新](context-attribute-actor-dp-fixed-image-cpu-v1.json)。
实际probe为`tools/reports/probe_context_attribute_actor_dp_cpu.py`，输入采用合成音频特征、
合成主干与真实共享Transformer head，未使用Qwen权重、LoRA或NPU；没有效果或容量结论。
复杂任务仍dispatch=false，余项为全量TRAIN processor终态与独立审核、selection实际输入
审核、真实音频Qwen/NPU的训推/更新/新进程恢复及最终不可变运行包。原简单曲线不受影响。


### 全选择集实际processor审核已启动

新增`context_attribute_selection_processor.py`以hash固定的TRAIN审核/独立核验源码为基础，
仅适配9226人口和selection来源字段，原TRAIN脚本不修改。选择集manifest、inputs、catalog
及audio元数据与冻结清单绑定；不向processor提供targets，不读取TEST，不计算模型指标。
4项本地检查通过，包括真实人口匹配、9226/40513混用拒绝和重新封存后错误文件的拒绝。

固定镜像52ba94e1容器2dd80642e9f148e040883a6436b97b696e28d63188efb69542b3f6be92a450a3
已启动，2CPU/8GB、无NPU/网络；1085包身份封存。先将校验hash的完整selection音频
物化到本地/data，Ceph音频只读；随后实际编码Direct/Q-input各9226次，校验相同波形、
声学特征、全部60/124读出和4096长度上限。结果保存张量身份记录，不另建大型特征缓存。

driver724207已运行并持准确docker wait，1092依赖身份封存；终态自动收集和独立核验。
见[启动证据](context-attribute-selection-processor-start-v1.json)。尚无完成收据，
不能称选择集processor已通过。原TRAIN driver704649与正式GRPO保持，三组1073/1078/1092
依赖校验均未变化。本步骤补齐新任务输入支持，不作为既有简单训练的额外门槛。


### 紧凑输入到actor的桥接：真实音频CPU检查通过

`tools/reports/context_attribute_input_candidate.py`复用既有不可变CPU缓存、音频读取和
online组装，独立绑定紧凑编译器hash、catalog与60/124读出。输入缓存身份区别于旧renderer，
不含policy、targets、奖励或算法；Q-input/Q-credit同输入共享缓存，但actor契约独立。
标签仅在消费时按input SHA及完整catalog绑定，按Final投影核验；engine只收到四项
无标签processor tensors和positions。在线组装沿用G16、固定Stage1reference与辅助衰减，
补入actor contract和节点顺序，不修改原online或训练实现。

固定52ba94e1 CPU容器4c3b976198dee58e已exit0/非OOM；392项包身份和产物独立收集，
12tests全部通过，无skip。使用既有最短/中位/最长真实TRAIN音频和实际processor：
冷/热缓存逐tensor与直接编码相同，浮点输入统一BF16；旧renderer缓存不命中，两个Q分支
可共用输入但不共用reward契约。三臂两次在线组装到真实verl DataProto通过，reference
复用、当前概率刷新及target/order/mask/padding/分支损坏拒绝均检查。
概率engine为替身，未Qwen forward、NPU/LoRA或新训练派发。
证据：[真实输入桥接检查](context-attribute-input-tools-fixed-image-cpu-v1.json)。

现有`FixedContextRollout`读出上限由max_decisions配置，`ContextProbabilityPooler`按
marker消费，并不硬编码60；可以复用现有推理工程。配置124不等于实际NPU容量/概率
一致性通过，仍需独立有限真实Qwen验证；不改变原策略工件契约来伪造新任务验收。
全TRAIN输入最新local1794708062.7012331仍running/nonOOM，4352/40513，
下一普通>=1794708662.7012331。选择集driver724207继续准确wait；无全量成功收据。


### 真实策略工件与HF/vLLM对齐入口已准备

新增`probe_context_attribute_parity.py`，复用固定Stage1策略/adapter、BF16 actor及
`FixedContextRollout`，HF/vLLM分进程执行，不复制推理引擎。配置固定三个TRAIN音频
锚点与Direct/Q两个模板，共六请求。保留Stage1原始工件契约，新的输入身份单独绑定；
Q-input/Q-credit输入相同，首轮共享Q对齐证据，不假称奖励训练也已通过。
比较器核验policy、输入panel、节点顺序、完整输入张量身份与全部概率，要求概率精确一致；
单独拒绝0/1饱和造成的非有限log策略。该有限诊断不替代全人口或DP更新/恢复验收。

固定52ba94e1 CPU容器199a7ba45ec1b427正常exit0/非OOM，404项包身份及产物独立收集，
9项比较器检查通过、无skip。真实完整Stage1 policy/adapter认证与六输入BF16准备完成；
独立复核六条input_ids身份及全部marker位置与既有实际processor锚点逐项一致。
见[固定镜像准备](context-attribute-parity-prepare-fixed-image-cpu-v1.json)及
[独立锚点匹配](context-attribute-parity-input-anchor-match-v1.json)。
没有模型前向、NPU对齐或训练收益，尚未派发NPU任务。

本轮实际npu-smi检查：175八卡有现存进程（0–3现有vLLM，4–7正式RL），181八卡也被
vLLM占用且固定52ba94e1镜像不存在。不抢占或并跑候选；设备空闲后再核实租约、部署
候选并执行分进程有限对齐。该资源条件不阻断原正式曲线或CPU输入审核。

selection实际审核local1794708505.3869793仍running/nonOOM，9226音频全部镜像且hash校验，
501346190字节；processor完成1280/9226，下一普通>=1794709105.3869793。
driver724207仍负责准确wait与终态收集，未有全量完成收据。


### Final/属性统一评测候选

新增`context_attribute_metrics_candidate.py`复用同一ContextEventMetrics，分别计算Final、
scenario、action集合F1、top1/macro-F1、NLL/Brier和空/多选率，并按原生结构歧义切片。
目标必须与输入SHA、catalog顺序、完整投影及来源组绑定，概率有限且完整；缺标拒绝。
Final直接由自己的概率读出，不用过程属性修正。Direct与Q的Final相同概率给相同指标。

完整11点曲线按Final set-F1最高→Final NLL最低→最早步选择，过程指标不参与。
NLL无限使用显式标志和null保存，不能当0或较优值；同时保留来源组/逐条记录以供
之后配对比较。当前paired_delta是描述性差值，不提供bootstrap或多种子结论。
评测内核验证传入人口，但不冒充真实推理认证；正式评分仍需发布/输入/实际模型产物审核。

10本地CPU测试通过；固定52ba94e1 CPU全9226选择集四组人工控制已启动，容器
40181840903c3424c5373c64b2fa8f4993885827e36cab013510007f72f7e84a，382包身份，2CPU/8GB，
无NPU/网络，准确wait81075等待。控制为全部正确、全不选、全选、仅过程正确，
分别按解析式独立核对三族F1和binary NLL，并核对2146来源组和切片人口。
这是人工概率验证，不是模型评测或复杂训练准入；终态待收集。


durable driver738521已持同一metrics CPU容器准确wait，终态后自动运行独立收集脚本；
不重复collector，不因观察尚未返回重启。训练1078、TRAIN审核1073、selection审核1092
冻结依赖本轮核验无变；最新TRAIN审核5888/40513，仍running/nonOOM。


### 全人口评分控制终态通过与配对不确定性口径

metrics CPU容器40181840903c3424已exit0/非OOM，driver738521自动收集完成，382项包身份、
10tests全部通过且无skip；四组人工控制全部9226录音/2146来源组通过。收集后另按解析式
复算三族F1/NLL与切片分母一致，见[固定镜像全量评分检查](context-attribute-metrics-tools-fixed-image-cpu-v1.json)。
“仅过程正确”控制的Final集合F1为0；说明评分没有属性修复，不是模型无收益结论。

新增`context_attribute_paired_bootstrap.py`，固定2000次、PCG64 seed317、95% percentile
区间和linear分位数。对同一来源组的所有录音整体有放回重采样，两策略共享抽样，
每次以实际抽中的录音数为分母；保持recording-weighted mean gain，而不是等来源组均值。
报告Final集合F1、精确集合、top1正确率的配对差及区间，不以区间选checkpoint。
step634是预登记主要端点；source-selected结果明确为给定选定模型的条件描述，
不声称对选择偏差、多重比较或训练随机种子不确定性作了校正。source_group也不是speaker。

9项本地测试通过，包括三个不等大小组全部27种抽样与独立逐条复制计算完全一致、
文件次序不变性、策略交换、零差控制及错误配对拒绝。固定镜像容器
3da0272b1ad804ddaf3b917a9f0166075543cc26ba48e09d64481684865b14ba已启动，381包身份，
实际全9226/2146人口的常数差控制也将验证，准确wait38277等待终态。仅评分工具，
不产生模型效果证据、不改变当前运行或排队包。

selection实际processor local1794709361.1312535仍running/nonOOM，2560/9226；
下一普通>=1794709961.1312535，原driver724207持准确wait，不重复派发。


bootstrap固定镜像3da0272b1ad804dd已exit0/nonOOM，381包身份与产物独立收集，9tests全部
通过无skip；全9226/2146来源组人工常数差控制及其95%区间独立核对通过。准确wait38277
已返回0。见[固定镜像配对区间检查](context-attribute-bootstrap-tools-fixed-image-cpu-v1.json)。
工具已具备，实际复杂任务模型预测和曲线尚不存在，不能报告真实性能差或置信区间。
当前训练1078、TRAIN审核1073、selection审核1092项冻结依赖均无变化。


## 六请求真实模型诊断封包（未启动）

原固定镜像CPU准备包404文件保持不变，新405文件包封存为
`.local/decision-objective-research/context-attribute-parity-npu-v1-prepared`。
独立进程依次执行HF/vLLM，复用既有六请求probe和Stage1策略；不改变阈值、概率精度或模型。
启动入口默认只做本地核验，显式派发要求已有简单基线/迁移队列完成、卡7空闲和同一租约，
并先保存attempt/容器身份再start，保留准确wait；失败产物也收集，不自动重建。
尚未派发，不是NPU概率/更新/恢复或复杂任务训练准入。实际结束后仍需独立审核容器配置、
输入和输出，不能把终态文件作为全部验收。

9项比较器和4项生命周期本地检查通过；后者使用进程/存储替身，仅覆盖顺序、失败归档、
拒绝覆盖和包污染拒绝。封包逐文件核验、配置404依赖、原CPU包身份均通过。
证据：`context-attribute-parity-npu-package-readiness-v1.json`。


## 诊断身份修订与Stage1结构结果

原六请求probe及405文件NPU v1包的请求哈希函数与实际FixedContextRollout不一致，
撤回其可派发资格，保留原始证据。独立v2工具/406文件包统一compact wire序列化；
全9226已归档实际请求通过v2身份校验，14本地测试含实际_score路径/engine替身。
未修改冻结src或原包，未新NPU推理。参见context-attribute-parity-wire-identity-fix-v2.json。

完整Stage1选择集按冻结结构分组分析已完成，原混淆及指标复算匹配。
1654 top1错误中540仅action错、144仅scenario错、970均错；754 top1正确但空集。
结构需组合组反而表现较好，不能称本体复杂度已代表模型难度；不改变任务/权重/阈值。
详见context-stage1-native-structure-diagnostic-v1.md及其逐记录JSON。


## 修订探针的固定镜像验收

v2 probe现已通过固定52ba94e1镜像CPU验证，运行包v4共409文件，14tests无失败/跳过；
六条真实TRAIN输入重新processor，token、mask、BF16特征及positions与原锚点完全一致。
NPU候选包中404项payload与已执行CPU包逐文件匹配，排除重定位config及未使用的旧CPU runner。
容器3c9a878eea5e9d941d01d63c954dc005ee77520aafed619e0bc87ffe79c36d1d exit0/nonOOM，独立收集完成。
证据context-attribute-parity-prepare-fixed-image-cpu-v4.json与context-attribute-parity-v2-fixed-image-input-match-v1.json。

第一次v2固定镜像测试因tests命名空间遮蔽在收集阶段exit2，未输入准备；v3尝试引用不存在的
本地tests/__init__.py而只在打包阶段失败、未派发。v4在包内新增空包标记，模型与测试代码不变。
完整失败证据保留。新NPU probe仍未派发；不以CPU通过代替真实Qwen概率、DP更新或恢复验收。


## 六请求的独立结果验收入口

`audit_context_attribute_parity_npu.py`完成，本地12项检查通过；采用真实封存包与CPU anchors，
模型概率、容器inspect及transport为替身。除后端间逐概率比较外，两端实际输入身份均须匹配
先前真实CPU processor的token/mask/BF16 feature/position，避免两端一致但共同消费错误输入。
同时绑定实际镜像/命令/env/device/只读挂载、完整传输hash及终态source manifest。
执行命令见`../engineering/context-attribute-parity-audit-runbook-v1.md`。
当前无实际NPU结果，审核器通过不等于模型验收通过；不修改已有实验准入或排队依赖。

## 复用现有日程与rollout会话的候选桥接

`tools/reports/context_attribute_schedule_candidate.py`复用既有缓存预取、reference缓存、
策略切换和采样生命周期，只替换属性请求、标签绑定及动作契约，不新建训练器。
6项新检查及5项原会话检查通过；输入编码和engine为替身，reference磁盘缓存与采样为实际路径。
新会话动作重放不能视为optimizer恢复，更不能代替真实Qwen/NPU连续更新验收。
收据见[桥接候选范围](context-attribute-schedule-session-candidate-v1.json)。

该候选尚不可训练。接入前需完成全量processor审核、紧凑输入的新长度分组身份、
绑定arm/catalog/objective的checkpoint契约，以及真实训推、连续更新和新进程恢复的有限验收。
这些只约束新复杂任务，不增加当前简单GRPO/RLOO的前置条件；原训练及审核依赖不变。

## 属性训练的日程与恢复契约

新增`tools/reports/context_attribute_training_contract.py`，复用原日程加载器、长度分组器和
checkpoint验证/恢复逻辑。原全量634步的query顺序、曝光及padding权重保持，仅根据
实际紧凑输入的token/frame长度重新分配rank；不直接复用旧renderer的分组身份。
发布入口要求完整TRAIN真实processor终态和独立审核，当前审核仍在运行，尚未发布实际新日程。
Q-input/Q-credit共享输入长度及顺序身份，但checkpoint独立绑定arm、catalog、design、
reward、概率数值协议及完整日程，不能互相恢复optimizer。

13项新检查及11项原日程/会话检查全部通过，共24项/13.77秒。使用真实634步顺序和合成
长度检验分配/曝光，CPU两参数Adam验证拒绝误恢复及同配置恢复后再更新相同；不是模型
恢复、新进程恢复或固定镜像/NPU验收。正式1078/TRAIN1073/selection1092依赖无变化。
证据见[训练契约候选](context-attribute-training-contract-candidate-v1.json)。
下一步接现有有限训练器的目标清单、日程与契约入口，再做真实模型有限验收；不新增训练器。

## 有限训练入口与完整selection审核修订

`context_attribute_training_candidate.py`以七处精确替换连接原有限DP4训练器，复用原
连续更新、collective恢复、BF16导出和engine激活代码。全40513条targets先与clean源
标签及音频身份逐条绑定；arm、reward、目录、数值协议、派生日程和适配源码共同进入
checkpoint契约。9项本地入口检查通过，其中runtime构造的日程manifest为替身；没有
执行真实Qwen/NPU训练。固定镜像CPU集成检查独立进行，不更改正式实验。
详见[有限训练桥接](context-attribute-trainer-bridge-candidate-v1.json)。

完整selection processor容器2dd80642已exit0/nonOOM，9226录音的18452个请求已收集。
旧独立审核将非整帧音频长度向上取整，实际Qwen单wave最长padding会丢弃最后部分hop，
需要floor(max(samples,8000)/160)。共有866录音/1732请求在旧公式下错一token。
新版`verify_context_attribute_population_v2.py`保留旧文件和失败收据，除修正取整外，
还独立生成每条完整int32 prefix mask并匹配原始tensor哈希，全部9226通过。
证据见[全量selection独立审核v2](context-attribute-selection-processor-independent-v2.json)。
这是输入审核，不是模型预测或效果；全量processor未重跑、训练数据未更改。

TRAIN原审核仍运行，保留原owner；待其终态收集后用同一v2规则独立复核。
新的派生日程入口为`prepare_context_attribute_schedule_v2.py`，使用修订审核而不改旧builder；
尚无完整TRAIN终态，不发布实际日程。固定镜像v2额外验证实际processor非整帧边界。

固定镜像v2现已exit0/nonOOM并独立收集，404文件包、46tests全部过无skip；1项依赖原源
绝对路径的完整runtime构造在容器中显式deselected，该项本地通过。三臂真实三音频global128
两会话更新覆盖BF16输入、原预取/reference缓存和native verl传输，engine概率是替身；
不是Qwen模型更新。实际processor五个边界长度的mask、feature宽度、audio token数精确匹配
floor公式，源码与既有同镜像归档逐字节相同。首次v1三项失败来自测试浮点边界断言，
原生产系数未调整。见[固定镜像桥接复验](context-attribute-training-tools-fixed-image-cpu-v2.json)
及[全量审核修订证据](context-attribute-frame-audit-resolution-v2.json)。

## 原输入任务结束后的自动收口与有限运行配置

本地driver792402持独立锁，等待原TRAIN collector锁释放，随后重验完整成功终态、v2审核、
三臂分桶日程和checkpoint契约。它不调用SSH、不收集远程文件、不启动训练；原owner704649
继续负责准确wait和收集。若旧审核因ceil公式失败，其证据保留，新后续只在完整成功transport
存在时继续。1167项依赖封存；输出为`.local/decision-objective-research/context-attribute-full-input-publication-v2`。
当前TRAIN未完成，输出尚未发布。

`context_attribute_execution_candidate.py`补齐两步工程验收的actor/service配置，复用原进程
监督器。Direct60/Q124与服务容量严格对应，恢复服务加载已提交step1策略，reference仍是
Stage1；恢复1→2与连续0→2需另取真实NPU证据。共享监督器、服务和训练入口源码按hash绑定。
37本地检查通过，包括真实子进程监督器回归；模型、checkpoint和设备未实际执行。
见[运行候选范围](context-attribute-execution-candidate-v1.json)。完整封包和NPU验收仍待，
正式GRPO/RLOO及迁移队列未变。

## 有限NPU验收源码封存

本地封包器`prepare_context_attribute_training_package.py`分为源码封存与完整输入发布后组装。
1158项源码/配置封存到`context-attribute-bounded-runtime-v2`，清单SHA为
`2558b169a61a69dc07ac59be0b98f471cc181e2832433a8337f5044835939b28`；隔离解释器导入、
原有限训练器及三臂actor生成通过。v1本地导入产生pycache导致精确文件集合校验拒绝，
保留现场；v2使用相同源payload并关闭字节码写入，未改变模型或训练逻辑。

完整封包要求已完成的真实TRAIN publication，再绑定Stage1策略、全量targets、三臂日程、
源文件和actor/service配置；Docker命令显式挂载4–7卡、只读共享模型/数据与publication依赖，
热缓存和输出使用本地/data。三个本地检查通过，但缺实际TRAIN终态，完整prepare成功路径
尚未执行；三臂计划不伪造publication身份，不自动派发。见
[源码封存证据](context-attribute-bounded-runtime-seal-v2.json)及
[封包准备范围](context-attribute-package-candidate-readiness-v1.json)。
