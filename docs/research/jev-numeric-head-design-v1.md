# Special-token 数值读出：LM head 与独立 decision head

状态：响应用户最新架构讨论的设计建议，未改变运行训练或宣布新 head 已支持 vLLM。
用户目标是可区分的数值判决输出，供 harness 消费，而非输出固定类别编号。

## 最新用户修订：历史上下文 Transformer 分支，最终输出 prob

本节优先于下文早期“先用线性头”的建议。用户要求决策分支可包含一层浅 Transformer，
读取过去历史上下文，最终对外仍输出概率。线性标量只作为末端投影或对照，不代表
主分支的全部计算。主设计候选为共享的、上下文条件化的 decision Transformer。

令 backbone 输出 `H=[h_1,...,h_L]`，决策 token 的位置为 `p_i`。一层分支可以按
稀疏 query 的 causal Transformer 实现：

```
u_i = h[p_i] + Attention(Q=Norm(h[p_i]), K=Norm(H), V=Norm(H), mask_i)
d_i = u_i + FFN(Norm(u_i))
z_i = w^T Norm(d_i) + b
prob_i = sigmoid(z_i)
```

这里 Attention 包含可训练 Q/K/V/O 投影与多头机制，FFN 也训练；不是对历史简单求均值。
mask_i 只允许当前请求的有效 token，且位置不晚于 p_i；padding/其他请求不可见。
所有候选完整定义和可用历史先于首个决策 token。最后一个综合决策 token 可以放在所有
配置问题之后，读取它们在 backbone 中的上下文表示。位置编码需与分支协议一起固定，
不能凭 backbone 已含位置信息就宣称任意历史顺序都可辨别。

与对 L 个位置都追加完整 self-attention block 相比，只有 K 个决策位置作为 query，
attention score 规模由 L×L 变为 K×L。它仍是上下文 attention+FFN 决策层；K/V 的
历史投影与 backbone 反向仍有成本，不能宣称总计算成本只有 K×L。先验证与同参数、
同 mask 的完整一层在选定位置的输出/梯度关系，再做 NPU 吞吐比较。

单层各决策 query 并行计算，因此 Final 看不到同层刚计算出的其他 d_i；它看到的是
backbone 的历史表示。这符合一次调用输出多 Span，但不证明新分支内部发生了串行
多步推理。若必须让 Final 消费新分支前置决策隐藏态，需显式增加第二层/分阶段连续
计算并验证收益；不偷偷回填离散预测或声称一层具备这种依赖。

prob 是唯一必需的对外数值。内部 z 用于稳定计算 logprob/loss，可写诊断但不冒称
对外成功概率。候选“适用性”用 sigmoid，多个候选可同时高概率；互斥选择任务使用
softmax(z/temperature)，其值是菜单条件化选择概率。两种语义分别登记，不强行统一。
需要 abstention/unknown 时须定义并监督对应事件，不能从低分自动伪造真值。

“历史”首先指当前输入中真实提供的音频、问题、规则和允许的对话历史。无历史标注的
单句 SLURP 不支持声称跨轮状态推理；更早请求不会自动留存在这个分支中。默认仍为
完整请求前向，跨请求 KV cache 属于另一个尚未验收的状态管理协议。

训练主候选：backbone LoRA + 浅 decision Transformer 全参数训练 + 小概率投影，
原 LM head 冻结。新增层规模、FFN宽度及位置编码先作为配置，不直接复制原 Qwen层；
初始化用小残差/可控门值，避免同时把两条残差和输出都置零造成无梯度启动。
门控为零时分支内部第一步可能无梯度，应在初始化测试中明确记录，不视为自动稳定。
专用 token embedding 训练范围、tokenizer 和分支全部进入策略快照与 vLLM加载协议。

当前 vLLM pooling 能取得完整请求 hidden chunks，具备实现此分支的数据入口；现有
pooler只执行LM head，尚未实现该Transformer、mask或权重切换。现有原头概率验收
不能复用于新分支。必须重新验收概率、梯度、batch/padding、恢复及热切换。

### 独立 CPU 实现与接口约定

`src/multimodal_jev/context_decision_head.py` 已实现上述单层稀疏 query 分支，
`forward(H, positions, valid_tokens)` 仅返回 `[batch, decisions]` 的 `prob`。
调用方将每个值绑定到请求内的问题/候选 ID；例如 `{"candidate_id":"route_a","prob":0.82}`。
这里的 0.82 是接口示例，不是实测结果。内部 `score_logits` 供稳定 loss/logprob 计算；
不以输出数值文本触发第二个生成步骤。每个 batch row 只允许一个完整请求，不支持
把多个请求拼接进同一行而不提供隔离协议。marker 必须在实际音频展开后定位。

候选配置为 `configs/experiments/context-decision-head-candidate-v1.json`：1024维、
8 heads、FFN1024、一层、残差系数0.1、无dropout；尚未经过NPU效率/效果选型。
分支不加新位置编码，使用backbone带位置的隐藏态，历史顺序能力需实际数据验证。
残差系数、结构和位置协议与权重一同构成模型配置，不能仅保存state_dict后换默认配置。

10项CPU测试通过：独立完整一层公式与稀疏查询的输出及参数/输入梯度一致；
历史敏感性、未来/padding/跨请求隔离；所有分支参数初始梯度非零；FP32/BF16
保存重载概率精确；错误位置和非有限输入拒绝。测试中的BCE仅用于梯度检查，
并未锁定正式训练目标。测试用合成隐藏态，尚未验证真实音频、NPU或vLLM。
sigmoid保证数值范围，但不自动保证概率校准；仍需Brier/NLL及可靠性评测。

专用token CPU探针v1终态exit1/non-OOM，根因是固定镜像tokenizer API不支持
`replace_additional_special_tokens`关键字，见
`docs/engineering/decision-token-registration-cpu-v1-failure.json`。尚未完成保存重载验证，
不能称新token已可用。naive新增ID可能占用timestamp预留151705的问题仍需解决。

## 当前实现的事实

`span_contract.encode_span_request` 将文本标记 `[JEV_SPAN_i]` 放在每个问题后，取标记
首 token 的前驱位置。该标记不是注册的单个 special token；因果模型的该读出没有看到
标记自身。`NativeAudioSpanModel` 在所取隐藏态上应用原 LM head，取 A/B 等符号的
logits 后归一化。当前 LoRA 导出仅支持语言 attention q/v，原 LM head 没有独立训练。

本地 Qwen3-ASR-0.6B 配置：hidden_size=1024、vocab_size=151936、
tie_word_embeddings=true。共享标量 `Linear(1024,1,bias=True)` 为 1025 参数；
完整词表矩阵为 155582464 元素，且与 embedding 共享。不能把解冻整个 LM head 当作
低成本替代，也不能把共享存储重复计为新增参数。
现有 `AudioSpanModel` 虽有独立分类 readout 原型，但不是已验收的数值 head 生命周期。

## 建议的目标架构

输入组织为：完整音频与状态/规则、完整候选定义、各判决问题及其 `<|jev_decision|>`
读出标记。每个候选对应一个判决位置；所有位置复用同一个小 head。一次 backbone
前向后收集标记自身的隐藏态，再批量调用 head，输出长度随候选数变化的数值数组。
没有条件启动第二个大模型或逐候选重跑 backbone，也不需要生成数值文本。

候选 ID 不作为输出参数索引；同一 head 从隐藏态读取音频、问题和定义关系。
所有完整候选定义先于首个读出，避免首个候选无法看到后续定义。普通 causal 模型仍有
位置/顺序效应；共享 head 不自动保证排列等变，需配对评测，必要时另立 mask 架构实验。
前置 Span 的连续隐藏表示可服务后续判决；离散预测不回填。

以下纯线性路线保留为历史设计与容量对照；当前主候选以最前面的浅Transformer分支为准。
只训新 head、完全冻结 backbone 可作为诊断，不能假设ASR隐藏态已具备规约消费能力。

## 数值的含义先于激活函数

首版推荐 `s_i = w^T h_i + b`、`q_i = sigmoid(s_i)`：
`q_i` 表示候选 i 在指定证据/定义下的适用概率，多个 q_i 不必和为 1；同时可返回 raw
logit s_i 作排序诊断。高分与低分可区分，但“间距越大越好”不是目标，饱和可能掩盖错误。
必须检验正负分离、概率质量和阈值下的风险。选择分布、适用性概率、预期效用分别定义。

若只用 pairwise loss，输出可解释为相对排序分数，不能自动称成功概率。
如果数值是预期效用，需定义奖励单位/范围并取得真实结果监督，不能用意图标签替代。
不确定性也不是任意追加一个 scalar 即可获得；缺证据/冲突需独立标签或明确估计方法。

## 三种头的比较与建议

| 路线 | 优点 | 局限 | 定位 |
| --- | --- | --- | --- |
| 原 LM head 冻结 + backbone LoRA | 复用已验收工程、符号先验；已有有限概率接口 | 依赖符号与词表读出，数值语义不直接；当前实现仍计算完整词表 | 保留工程与效果对照 |
| LM head 本身也加 LoRA | 可调整词表读出，同时保留生成接口 | tied embeddings/输出挂载、导出和推理支持都需验收；参数多于标量头，仍依赖符号 | 非首选，不把现有 q/v LoRA 误称此方案 |
| 独立共享标量 head + backbone LoRA | 输出语义明确、任意候选数、小参数量、摆脱 A/B 映射 | 新初始化和训推/恢复/热切换需支持；未证明效果更好 | 历史建议，现作为浅Transformer分支对照 |

不为每个意图建一个独立参数头。那会重新引入固定类别表，妨碍配置化迁移。
head 可按输出类型共享，例如适用性、偏好、证据状态，而不是按数据集或类别复制。

## 可验证的初始化桥梁

在同一隐藏态 h、原 LM head 为仿射投影且没有额外 logits 变换时：

`softmax([z_false,z_true])_true = sigmoid(z_true-z_false)`。

所以可用 `w = W_true-W_false`、`b=b_true-b_false` 初始化独立标量 head。
这只在同一读出位置/输入和实数运算下等价，不保证 BF16 算术逐位一致。
新 token 会改变输入、位置和隐藏态；将前驱读出改到 marker 自身也会改变隐藏态。
必须先在旧位置验证读出替换，再独立测试新 token/位置，不能一次改两项后宣称等价迁移。
符号差初始化和小随机初始化均是候选，不能未经模型验证认定前者最优。

新增 token 应固定 tokenizer revision、ID、embedding 行初始化和训练方式，核查模型预留
特殊 ID 的用途；不随意挪用 audio/control token。若扩词表，HF/vLLM 的词表、embedding
尺寸和共享权重关系必须一致。优先明确保存新增行，避免为一行训练整个 embedding。

## verl/vLLM 与 RL 接入要求

### 本地权重的初始化诊断

已只读提取本地共享 embedding 中 A/B 两行（token ID 32/33），对 128 个固定 seed317、
单位 RMS 的合成隐藏态验证同位置替换。FP64 最大概率误差 4.44e-16、隐藏梯度误差
3.52e-19；FP32 最大概率误差 3.58e-7；BF16 为 0.00210547。三种精度在此次 0.5
阈值下均无判决差异，序列化重载输出精确一致。见
[原始结果](../engineering/numeric-head-real-row-initialization-v1.json)。
这不是音频前向或 NPU 验收，合成隐藏态不能证明实际模型等价。真实音频 CPU 探针
另行比较原词表投影、保留 A/B 两行的小头和差分标量头，分离投影形状与权重作差误差。
两行小头同样可输出连续的 `z_true-z_false`，可作为迁移桥梁，不限定最终必须单行。

本地 tokenizer 最高已登记 ID 为 151704，模型词表行数为 151936，无已登记 Jev token。
这个差额不自动证明额外行可随意占用；必须核查 processor、tokenizer 新增规则和模型
控制 token，固定新映射后再测试。不能因为有预留行就未经验证宣称不需扩展/改动权重。

- head 执行/导出与 backbone/LoRA 使用 BF16；归一化和 loss 用 FP32，优化器 master/
  Adam 保留 FP32。不同算子求和次序的误差必须实际测量。
- head、tokenizer、token embedding、LoRA、读出位置协议组成一个策略版本；旧 logp、
  reference、rollout、恢复检查都绑定此版本。当前仅 q/v LoRA 导出不包含新 head。
- vLLM pooling 路径直接调用相同数值 head；需要显式权重加载及版本切换。
  不假定任意 `modules_to_save` 自动支持 vLLM 热更新。先建立固定版本推理，再验收
  整套策略切换；不能出现新 LoRA 搭配旧 head。
- 若 RL 动作为候选适用集合，可用 Bernoulli 策略的采样及 logp，但独立 Bernoulli 的
  联合分布假设须显式记录；边际适用概率不能直接当成真实联合概率。
- 若 RL 动作为最终单选，则另定义 candidate categorical policy；从 s_i 构造选择分布
  是一种策略设计，不能将 softmax(s) 与独立 q_i 的含义混用。
- 连续 s_i 本身不是 PPO/GRPO 所需的离散动作概率；需要先锁定动作、采样分布和
  credit/归一化方式。推理可以输出连续数值，训练的策略空间仍需精确定义。

监督损失可含 proper scoring、有效/无效候选对比和等价规约一致性；head 的改变不
自动决定采用哪种 loss，也不取消用户要求的两阶段和衰减监督辅助。新正式训练前先
完成任务与动作契约，再实现、锁定工程和超参。

## 对照顺序

1. 相同输入/旧读出位置，比较原符号 logit 差与初始化标量 head，验证数值和梯度。
2. 固定音频与候选定义，单独加入专用 token 和 marker 自身读出；检验长度、位置、
   初始化敏感性，不能将纯 head 收益与输入变化混为一项。
3. 在相同数据、预算和 RL 起点协议下比较两种 head 的完整曲线，报告适用性集合指标、
   分离度/排序、Brier/NLL、规则变化、候选换序与 FSC 冻结迁移。
4. actor/rollout 完整概率、真实更新、保存恢复、策略切换与下一步更新全部验收后，
   才将新路径视为稳定工程基线。已有单选 Stage1 独立保留原实验身份。

## 真实音频读出与token探针实证（2026-11-14）

`numeric-head-real-audio-cpu-v1.json`已收集固定镜像CPU成功终态，并独立重算两个
真实音频的隐藏态快照：原LM头与两行子头概率精确一致；合并成BF16差分标量后，
概率最大绝对差分别0.01181835与0.00460073，本次没有0.5阈值翻转。FP64代数
概率最大误差1.11e-16、对隐藏态梯度最大误差6.94e-18，说明代数等价不能代替
BF16执行一致性。该结果不代表新Transformer分支效果或概率校准。

`decision-token-registration-cpu-v3.json`已收集exit0/nonOOM与完整词表/原added
token属性独立JSON复核。直接追加ID151705会碰撞config的timestamp_token_id；
测试占位后decision ID151706在原embedding范围内，实际音频两marker位置77/92，
保存重载输入张量一致。占位token不是正式timestamp语义方案，不直接发布生产。
v2因逐词重复获取完整词表导致平方级工作量，已主动停止并归档，exit137是主动停止，
不是OOM；v3改为一次获取后逐项比较，不削弱检查。

`probe_context_decision_audio.py`开始真实音频单次backbone+浅Transformer验证。
三问为无监督结构探针，完全给定，不看gold；任意诊断线性泛函只检查LoRA/分支
梯度，不做优化器更新。测试token行使用A/B embedding均值并冻结，不宣称正式
初始化方案；分支保存重载、完整隐藏态/输出快照待终态复算。此探针尚无完成结果。

## 联合策略快照与有限动作接口候选

`context_policy_snapshot.py`将BF16浅Transformer参数、骨干LoRA、冻结marker行、
完整processor文件和分支/输入/运行协议共同纳入策略ID。读取时先校验完整文件集合、
全部hash、预期base/策略ID、结构、marker和每个张量，再写入目标模型；不把LoRA
文件ID当作新分支完整策略ID。调用方仍须验证实际加载base与传入base身份一致。
发布只允许经本地存储准入的目录，保持fsync和独立最终目录，不改原模型资产。

首版明确只做execution快照：**不恢复FP32 master/Adam**；仅允许冻结embedding
的指定marker行，尚未实现可训练行delta。新分支真实NPU/verl恢复、vLLM processor/
分支切换均未接入，策略文件通过不能冒称这些链路已完成。

`context_binary_policy.py`将每个logit转换为稳定的[false,true] logprob，供已有有限
Span actor采样/计算joint logprob。对外prob仍为sigmoid。RL探索在此显式选择因子化
Bernoulli策略，这是一种采样参数化，不声称语义适用事件独立，也不将乘积作为校准
的整集合真值概率。原GRPO/RLOO估计器与joint PPO clipping不改。不同实际问题数
须分组，不能补虚拟动作；缺少监督的真实问题仍是动作，reward/loss按监督mask。
此适配尚未接NPU训练，也没有冻结新任务奖励或辅助对比loss权重。

新增测试覆盖完整动作枚举概率与score-function梯度、BF16极值logprob、采样重放、
候选换序、联合权重恢复概率精确、未导出可训练参数拒绝及坏文件/processor/结构/
策略身份在修改参数前拒绝。连同head/contract共31项本地CPU通过；不是模型效果。

### 真实分支终态补充

`context-decision-real-audio-cpu-v1.json`：固定镜像容器0360ca92已exit0/nonOOM。
中位/最长两条实际音频分别239/509 token，均一次骨干调用得到三个prob；没有
ASR转写或生成回填。保存的真实hidden、branch权重和概率已本地独立复算，
BF16输出精确重现，分支和hidden梯度有限。首条远程诊断反向的16个head参数和
56个LoRA-B参数非零，56个LoRA-A梯度为零（新LoRA-B零初始化的预期结果）。
本地仅复算head/hidden梯度；未独立重跑完整骨干反向，不夸大验证范围。

探针不使用任务标签、不执行optimizer，也未训练marker行。初始化随机head输出
不能作为任务指标或校准结果。新联合快照模块尚未用于这个探针，只验证了branch
自身重载；完整Qwen策略的联合加载、FP32master/Adam恢复与vLLM仍待接入。
固定镜像快照/二元接口回归v2共32项通过，非有限marker拒绝也已覆盖。

### LoRA执行语义与完整Qwen恢复探针

联合快照新增逐模块LoRA r/alpha/scaling、矩阵形状、fan_in_fan_out、活动adapter、
未合并状态、禁用DoRA及零dropout绑定。仅保存矩阵和base ID不足以确定执行策略。
同形状/同权重但scaling改变的反例已覆盖；本地及固定镜像v3共33项测试通过。
首版仍只接受单default adapter及冻结marker行，不放宽为通用PEFT导出。

真实Qwen联合恢复探针4a0de48d已启动CPU：从之前实际音频参考构建完整策略快照，
扰动LoRA/head/marker后联合恢复，并用快照processor重编相同音频，比对旧结果。
尚未取得终态；无optimizer step/恢复、NPU或vLLM准入。

完整Qwen联合恢复探针4a0de48d已exit0/nonOOM，129个BF16张量（112LoRA、
16head、1marker行）与processor/执行协议共同导出；扰动后恢复张量精确，两条
真实音频重编及输出与此前结果精确一致。完整文件hash/策略ID和记录概率已独立
复核。证据`context-policy-reload-audio-cpu-v1.json`。该策略是未训练的fresh LoRA，
不涉及FP32master/Adam恢复或NPU/vLLM；不能用此结果代替非零训练LoRA的后端验收。
