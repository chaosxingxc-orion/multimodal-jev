# 语音 Jev 数据集与指标工作基线 v1

日期：2026-11-11。覆盖已登记的 **16 项资源**；定位为数据制作、实验设计和论文横向比较的共同指南。
本文综合仓库已有一手文献锁定、代码审计和本地验收，未重新检索全部在线版本、下载数据或启动实验。
论文/代码/权重 revision、表号及来源哈希继承 [学术登记](academic-baseline-lock.json) 与
[来源记录](academic-baseline-sources-v1.json)；不能据此声称穷尽当日 SOTA。

**已经确定的执行方向**：优先清理后的 SLURP，准备 Qwen3-ASR 端到端训练；FSC 作为简单控制；停止扩展 Omni 验证。
actor 直接消费音频、任务定义和允许的上下文，不使用参考转写，也不走 ASR→独立文本模型串联。
**本文建议、尚未获得用户最终确认的研究边界**：Jev 主任务为定义驱动的意图/函数选择和槽位类型存在性；开放值抽取、完整调用生成与实时执行分别建表。
这是一份可以审阅的工作基线，不能当作新训练发布、启动凭证或所有任务已实现的声明。

## 1. 论文要回答的问题及模型边界

建议主问题：**同一个语音模型能否根据动态定义作出正确、稳定且能区分规则变化的决策，并迁移到未见来源、规则和任务？**
仅把标签换成 JSON，或把 55 个槽位类型展开成大量二分类，都不足以回答这个问题。

| 能力 | Jev 适合承担的部分 | 需独立能力/监督的部分 | 可以声称什么 |
|---|---|---|---|
| 意图识别 | 单意图用有限 Choice；多意图用多个 Noul 命题或显式集合输出 | 开放意图描述、自由解释 | 定义条件下的分类/多标签决策 |
| 槽位类型 | 判断某语义角色是否有证据；多个类型可同时成立 | 开放姓名、地址、歌曲名的值与次数 | 槽位类型检测；不能称完整 slot filling |
| 槽位值 | 固定且完备枚举可用 Choice；给定候选的证据核验可用 Noul | 任意值、多次提及、嵌套结构用抽取/生成；时间定位需对齐监督 | 有限值选择与开放值抽取分表；不能用 gold 造候选 |
| 工具路由 | 哪些函数适用；参数是否有已标注证据 | 参数填充、调用次数及顺序、执行结果处理 | 函数适用性不等于已具备执行条件 |
| 对话控制 | 有标注时决定澄清、等待、取消、交接 | 自由回复、摘要、真实环境副作用、时序策略 | 静态选择不等于实时 Agent 成功 |
| 等级判断 | 真实序数标签或完整可计算业务规则对应的 Score | 主观虚构的紧急度/质量 1–5 分 | 不把类别编号或模型置信度当 Score 金标 |

例如“明天给张三打电话”，Jev 可以判断电话意图及人名、时间类型是否出现；“张三”的准确字符串、重复同名对象及日期归一化是其他能力。
同一音频骨干可以同时提供决策与生成读出；这个拆分不要求建立转写串联。
`open_slots`、`typed_call_list`、`typed_tool_sequence` 是项目扩展任务，不因有结构化输出就成为 NGA 的原始 Choice/Noul/Score。

## 2. 数据集组合与使用角色

下表是推荐组合，不要求把所有资源都加入训练或首轮全量运行。“外部评测”不等于模型预训练绝无污染；本项目既有研究/提示暴露仍需披露。

| 资源 | 建议角色与独特价值 | Jev 核心输出 | 独立保留的完整任务/主指标 | 当前关键门槛 |
|---|---|---|---|---|
| **SLURP clean-v7** | 主训练、机制消融、来源/规则泛化 | 意图 Choice；类型存在性 Noul | joint intent；类型集合 EM/F1；完整抽取另报官方 SLU-F1 | 语义角色定义、规则族划分、可训练发布未验收 |
| **FSC** | 轻量控制与回归验证，不作主要创新证据 | 31 合法组合 Choice；定义/拒识对照 | action/object/location 全对率 | 已有完整实测；任务简单且近饱和 |
| **MAC-SLU** | 优先中文、多意图组合外部验证 | 意图集合；意图绑定的类型集合 | 官方 Overall、intent/slot 指标；完整绑定 EM | TTS；adapter/绑定转换需验收；不混成单意图 |
| **SFC / SpokenFC** | 优先动态函数与 ID/OOD 迁移 | 每个函数适用性；有依据的参数存在性 | strict 调用多重集 EM、绑定参数 F1；官方 Overall 另列 | 保留两段音频、历史、NAN、数组；暴露条件分轨 |
| **Audio2Tool** | 只作复杂工具外部 TEST | 工具集合/存在性 | 分 tier 工具指标；完整参数可用子集严格 EM | 无 TRAIN；评分版本差异、附加参数缺标、输入成本高 |
| **SLUE Phase 1** | 自然语音类型检测与情感分类 | NER 类型存在性；情感类别 | NER phrase+tag F1；情感 macro recall/F1 | gated 音频未齐；类型检测不等于实体识别 |
| **SLUE Phase 2** | 优先对话行为 DAC；定位作扩展 | DAC 类别；有真实候选时证据选择 | DAC macro-F1；NEL/QA 时间指标；摘要指标另表 | NEL 依赖 Phase1；各任务独立 adapter |
| **MINDS-14** | 小规模多语言意图迁移 | 14 类意图 Choice | 每语言 accuracy/macro-F1、语言等权均值 | 无已核官方 HF test；需冻结自有划分 |
| **SpokenWOZ** | 多轮状态决策边界验证 | 槽位是否需更新、有限状态/动作（须可推导标签） | 状态值 JGA；回复 Inform/Success/BLEU 另列 | 整会话隔离；开放状态值仍需抽取 |
| **MMAR** | 直接对照 AudioJev 的外部 Choice 评测 | 原始音频问题 Choice | publisher accuracy + 严格匹配 | 异常候选保留；官方 CUDA 复测未完成 |
| **MMSU** | 检查语音/声学证据范围 | 有候选的原生任务 Choice | 官方聚合、逐任务成绩、任务等权补充 | full 与 short 不同；ASR 骨干非语音能力待验证 |
| **MMAU** | 语音、音乐、环境声外部泛化 | 原生 MCQ Choice | 官方 accuracy，三类别及 mini/hidden 分表 | release/隐藏评测/已披露音源重叠 |
| **VoiceBench** | 自由语音助手能力外围诊断 | 仅原生有限决策子任务 | QA、IFEval、开放 judge、鲁棒性分别计分 | 不能把自由生成直接改成 Choice 后沿用原分数 |
| **FDB-v3** | 后期真人语音工具与交互验证 | 有标签的动作/时机决策 | Pass@1、tool F1、argument、turn-taking、latency | 100 例、mock tools；provider/judge 尚需配置 |
| **τ² voice-v2** | 后期环境闭环主验证 | 策略/工具/澄清节点 | 官方任务完成、约束、交互质量与成本 | 仅环境桥有证据；没有模型语音闭环成绩 |
| **EVA-Bench** | 后期企业场景可靠性补充 | 有明确监督的策略节点 | EVA-A/EVA-X；pass@1、pass@k、pass^k | 本地 airline 不代表三域；外部服务依赖 |

首轮论文核心证据建议为 **SLURP + FSC 控制 + MAC-SLU + SFC**；Audio2Tool 提供更复杂的只测不训迁移。
SLUE 的自然语音价值很高，数据授权和适配满足后优先加入；MINDS-14 回答跨语言问题。
MMAR 是与 AudioJev 最直接的算法比较，但不能取代 SLU。MMSU/MMAU 的音乐、环境声结果应作为能力边界，不能预设 ASR 骨干能覆盖。
交互集在静态决策可靠性证明后接入，不让服务依赖阻塞 SLURP 首轮训练准备。

## 3. 各数据集的制作与横比细则

### 3.1 SLURP：主线及标签口径

clean-v7 TRAIN 保留 11,460 个文本来源、50,388 个录音引用、50,387 个唯一音频；额外隔离一条存在相互冲突标签的录音后，拟可用 50,386 个。
已有来源闭包划分：learning 可用 40,713，selection 可用 9,673；没有说话人隔离或语义近重复完全隔离的证明。
clean TEST 为 12,991 个录音，devel 为 8,690。依据：[清洗对齐](../engineering/slurp-clean-v7-alignment-v1.json)、
[TRAIN 审计](../engineering/slurp-clean-train-readiness-v1.json)、[来源划分](../engineering/slurp-clean-source-split-v1-verified.json)。

**标签必须写清字段**：本轮只读清洗 TRAIN 再计数确认，原始 `intent` 有 **91** 个名称，`(scenario, action)` 有 **60** 个组合，实体类型 **55** 个。
当前原生接口及拟议 Choice 使用后者；不能把“60 个组合”写成原始 91 类 intent 的同一任务。
评测沿固定原生映射；不从 TEST 扩充训练定义，也不构造 scenario×action 笛卡尔积。

实体类型根据语义角色和发布标注约定判断；同一词在不同上下文可属不同类型，不用字符串字典直接生成标签。
未标注类型只有在该任务标签可视为完备、定义无冲突时才能生成 false；否则标 unavailable/需复核。
重复同类型实体在类型检测中折叠为集合，在完整抽取中保留次数。原始 TRAIN 中的合成来源与 dev/test 重叠风险未隔离前，不整包混入 real-only 实验。

原始未经清洗 TEST 的已完成 Omni 成绩只作诊断：joint intent 39.9067%、SLU-F1 0.3141268，不能用于优化或与 clean 分数计算算法提升。
[运行凭证](../../experiments/slurp-omni-native-test-v1.md)。作者报告的 SpeechBrain HuBERT 75.10 SLU-F1 / 87.54% intent 为 real+synth 条件；HerMiT 70.84 SLU-F1 为历史串联条件。
它们只进入原生参考表；比较 clean 条件时需要在同一 clean 面板重评，不能拿旧论文分数直接排名。

### 3.2 FSC：保留简单任务的诊断价值

官方 train/valid/test 为 23,132 / 3,118 / 3,793，31 个组合意图。
FSC 的 action/object/location 是有限标签，并不提供开放槽值抽取难度；固定 `none` 类别也不是缺标或 Jev 拒答。
使用原始三元组全对率；三分量成绩、排列/符号重绑定/候选缺失、正反命题配对作为诊断。
候选缺失只能证明菜单外拒识，不能证明识别真实未知意图或听不清音频。

学术参考 Lugosch 冻结预训练声学模块 98.8%；SpeechBrain 官方报告 99.60%，本地公开任务权重复测 **99.4991%**。
本地 frozen Omni 严格候选接口父提示→教师 p1 为 **46.1904%→91.7743%**，但六视图全对仅 **0.2373%→0.6327%**。
这些是不同训练/接口条件，不能合成同条件排行榜。[完整结果](../../experiments/fsc-omni-official-test-v2-comparison.md)。
不再以 FSC 先导训练阻塞 SLURP；已看过的 TEST 结果不能继续用于本轮提示选择。

### 3.3 MAC-SLU：最有价值的中文多意图扩展

原论文 train/dev/test 为 17,997 / 1,391 / 1,151 个文本记录；中文车载命令、TTS 语音，多意图可共同出现。
请求保留音频与完整意图/槽位定义；target 保留 **意图实例—槽位角色—值** 的绑定。
决策投影至少保留意图集合及 `(intent, slot_type)` 关系；只把全句所有槽位类型取并集会丢掉“哪个意图需要哪个参数”。
若同意图重复出现且原标注无法唯一对齐，不猜实例匹配，另报可判定关系集合与绑定覆盖。

主报多意图集合 EM、micro/macro-F1、意图绑定的类型关系 F1、整句决策全对率；完整生成另用官方 Overall、slot F1 和严格绑定 EM。
既有 Qwen2.5-Omni-3B SFT **Overall 56.56** 是完整任务作者报告，不能拿类型 F1 超过它就声称超越。
[论文 Table 6](https://arxiv.org/html/2512.01603v1)、[固定官方代码](https://github.com/Gatsby-web/MAC_SLU/tree/d54453f63e0dc57b17d3cb2cfea35fb7da6a4489)。

优先作为 SLURP-only 模型的中文迁移；若后续使用 MAC-SLU TRAIN 微调，另立 supervised-adaptation 表。
同源文本的所有 TTS 版本同侧；不能仅依据音频 hash 不同宣称内容隔离，TTS 成绩也不证明真人中文鲁棒性。

### 3.4 SFC / SpokenFC：动态定义、函数 OOD 和多轮证据

固定来源 `QwenAudio/FunResearch@5cd9dc679584ba9b14454908a0f2c85e74c00147`；处理后 train-ID/test-ID/test-OOD 为 **5,534 / 2,590 / 771 turns**，对应 **4,499 / 1,986 / 593 conversations**。
多函数可同时触发，适用性用 Noul/集合；完整调用保留函数—参数绑定、数组、重复次数和官方缺参字符串 `NAN`。
第二轮保留前后两段音频及数据提供的助手历史；这是 supplied-history 离线任务，不是模型自主完成澄清。
存在性决策不自动提供“现在应执行”“应追问”标签，后两者需按任务规则和参数状态另行标注。

决策主指标为函数集合 EM、active precision/recall/F1、balanced accuracy、多轮 both-correct；完整任务另报 strict call-multiset EM 和 function-bound slot F1。
官方 Overall/函数/槽位指标保留独立列，已知 parser 对数组、参数归属、重复调用处理有缺陷，不静默替换。
已有 learning 审计中 inactive 占 **73.6238%**，全空策略获得同等比例旧奖励却完成 **0/4,428** 个任务，见
[奖励退化分析](../engineering/sfc-reward-balance-numerical-v1.md)。不能以函数逐行 accuracy 为论文主要结论。

必须区分三种训练暴露：SLURP-only transfer；SFC TRAIN 提示/教师适配；SFC TRAIN 权重适配。
后两者都不能写成 SFC 零样本迁移。现有 SFC learning/selection 准备及历史暴露见
[提示划分](../engineering/sfc-jev-prompt-partition-v3.json)；新 ASR 主线需独立登记采用哪一轨，不能继承旧提示后声称未接触 SFC。
数据没有 speaker ID，不能认证 speaker-disjoint。[官方/strict 协议与学术分数](07-academic-baseline-lock.md)。
论文 Table 5 基座 49.03 与 Table 7 基座 48.80 设置未完全对齐；SFT/GRPO task checkpoint 未核验，不据此制造受控增益。

### 3.5 Audio2Tool：高难迁移，但必须防止负例和缺标造成虚高

全部固定为外部 TEST。已验发布有 **36,419 WAV**；本地输入/目标协议为 **30,733 发布记录、30,731 唯一输入**。
这些与卡片 16,843 queries、36,421 音频及论文约 30k 的口径不同，分别注明；录音、query、记录、决策不是一个分母。
工具来自独立 **152 项 registry**，不能从 gold 工具名/参数生成候选；多轮保序，音频变体不当成对话轮次。

当前 Jev 每条 457 项决策是旧发布设计，**不作为必须照搬的训练/推理形态**。
完整恒不调用控制的 Noul accuracy **99.1925%**，必需工具召回 **0**，工具序列 exact **0**。
所以主报按 tier 的工具集合 EM、active P/R/F1、balanced accuracy；正反命题同时正确只作补充。
完整展开约 1,404 万决策、逻辑输入约 982 亿 token，不能用“一个小头”推断整体便宜；新共享编码/批量读出需单独测成本。

25,969 条记录的完整调用参数可投影；4,764 条缺少完整附加参数，不能将缺标填 `{}`、`[]` 或 null 当正确答案。
全量工具名与已知主参数可以计分；完整参数 EM 仅对明确可用子集报告，**全量完整参数 EM 未定义**。
监督缺失与模型失败分开：前者登记不可评分覆盖，后者占可评分全分母。

论文定义的 ordered tool sequence/micro slot F1，与当前官方代码的工具集合/逐样本参数 F1/排除错误行不同。
同时固定“上游函数+本地全分母”指标和本地 strict 指标，名称注明 scorer；不声称完整复现论文。
主调用+附加调用的标注顺序也不自动等于可执行时间顺序。
作者 Qwen2.5-Omni-7B 的 Tier1 tool Acc **79.5**、Tier2 EM **14.7** 只能在对应原生条件参考。
依据：[官方代码审计](audio2tool-official-code-audit-v1.md)、[评分契约](../engineering/audio2tool-score-contract-v1.md)、
[全量 Jev 与退化控制](../engineering/audio2tool-jev-v1.md)。

### 3.6 SLUE Phase 1 / Phase 2：自然语音及任务边界

Phase1 NER 将全句 entity-type 集合用于 Jev 时，主报 type-set EM、micro/macro-F1；原生实体 phrase+tag micro-F1 和官方 label-F1 独立列。
**自定义类型集合 F1 不命名为官方 label-F1**：官方匹配单位/计数需沿锁定 scorer。
情感适合原始类别 Choice，主报 macro recall/F1，不能把正/中/负任意改成连续 Score。
VoxCeleb 使用已锁 v0.2 分布修订；fine-tune/dev/test 原分区保留。当前最后资产证据显示 Phase1 gated 音频未齐，不将计划写成可训练。
[Phase1 论文](https://arxiv.org/html/2111.10367v2)、[固定 toolkit](https://github.com/asappresearch/slue-toolkit/tree/ed23039507c8f01b704a5c43ed89b6e808a49405)。

Phase2 分四个独立任务：DAC 对话行为最适合第一轮决策扩展，主报官方 macro-F1；多标签与否以固定发布的标签契约为准。
NEL 是实体时间定位，用 frame-F1/按 overlap 阈值的 word-F1；类型检测不能替代它。
QA 保留官方音频问题/文档角色及时间答案，使用 frame-F1；SUMM 是自由摘要，用 ROUGE/METEOR/BERTScore，不强行塞入 Jev 主表。
NEL 训练复用 Phase1 NER，Phase2 归档已下载不解决该授权依赖。
[Phase2 v2 论文](https://arxiv.org/html/2212.10525v2)：DAC 管线 macro-F1 **70.7**；QA 管线/E2E-DUAL frame-F1 **39.6/21.8**；NEL E2E-w2v2 test frame-F1 **56.3**。
这些是不同任务及输入条件的作者报告，不能平均或跨列排名。管线仅文献参考，不派发串联路线。

### 3.7 MINDS-14：跨语言小样本诊断

14 类银行意图适合 Choice；保留语言/方言标签与原音频，不让语种决定候选子集。
HF 主要提供 train，原论文 60/40 随机划分及三次评估的 fold IDs 尚未核验。
先冻结自有来源分组的 learning/selection/test 名单，明确称 project split；来源同义句/跨语言翻译有关系时连接隔离，不能随机切衍生行。
每语言报 accuracy、macro-F1、样本数和区间，跨语言用等权均值补充，避免大语言分片主导。
若使用任何 MINDS TRAIN 适配，就改称监督多语言轨道。

[原论文](https://arxiv.org/pdf/2104.08524v1) 管线 **93.4%** 与 AudioJev 的 **672 条、六语言变体、88.59±0.23%** 不是同一评测集。
不直接比较，不以本文选定划分宣称复现原论文。小数据更适合验证迁移方向，不作为宽域多语言能力的唯一证明。

### 3.8 SpokenWOZ：对话状态不是槽位类型的同义词

5,700 真人对话、约 203k turns，按完整 conversation 分区；所有历史轮次同侧。
输入保留当前及需要的历史音频/允许系统上下文，参考转写和 gold state 不进入 actor。
若研究状态更新，应区分沿用、覆盖、删除、dontcare、未提及等原生语义；只有定义完整可推导的部分才生成 Jev 标签。
“这轮提及 hotel-area”与“对话结束时 hotel-area 的值正确”是两个任务。

决策子表报槽位更新类型 F1/集合 EM，原生主表仍报 **JGA（当前全部状态值全对）**；逐槽 accuracy 易受未改变/空槽支配。
完整状态追踪需要开放值与跨轮绑定模块，决策模型不凭类型分数声称完成 DST。
[论文 Table 7](https://arxiv.org/html/2305.13040v2) SPACE+WavLM_align **JGA 25.65** 为音频+文本条件；不得当纯音频同条件对照。
fixed test 与论文版本需对齐；离线 Inform/Success/BLEU 不等于实际工具环境任务成功。

### 3.9 MMAR、MMSU、MMAU：Choice 泛化与音频能力范围

**MMAR**：1,000 题保持全音频和原始候选顺序；4 条含空候选、6 条多 reference，其中 5 条在固定单候选 publisher 契约下无可满足答案，均保留并标明。
publisher word-set 判定与严格候选匹配并报，不修 gold 或只挑可答题。
[AudioJev](https://arxiv.org/html/2610.01293v1) 三 seed **55.33±0.84%**、公开单权重 **56.30%**、Frozen Omni3B **53.50%** 分开登记；原 MMAR Qwen2-Audio **30.0%** 是外部生成式参考。
本地没有已完成的 AudioJev 模型复测；CUDA backend 移到 NPU 必须标 port 并验证。[本地协议](../../experiments/audiojev-mmar.md)。

**MMSU**：full 5,000/47 tasks 与 AudioJev short 3,931/39 families 独立。
官方 Qwen2.5-Omni-7B **60.57** 与 AudioJev short **62.45** 不能作 full 提升。
按任务、语音/韵律/副语言等类别报告，官方聚合与项目任务等权均值分别命名。
长音频不静默截断；不支持的样本留在预定评测覆盖表中，显式失败计入可评分分母。
[官方基准](https://github.com/dingdongwang/mmsu/tree/217d1d3f5ad59203ed16f44c0fd913f4f8add2b4)。

**MMAU**：固定 `MMAU-v05.15.25`，mini 1,000、hidden 9,000、full 10,000 独立。
mini 是公开测试集，不能默认为调参集；hidden 使用官方服务，先锁模型/提示。
语音、音乐、环境声分列；AudioJev 单权重 full **69.41** 与三 seed **68.88±0.46** 分列。
作者披露 119 个 hidden 音源与拟合/开发材料重叠，报告暴露，不能声称完全无污染。
[固定官方代码](https://github.com/Sakshi113/MMAU/tree/110127f54c0dfba3faa5ec9feee4a7e4148679c5)。

这三者均不用于本轮教师反馈、奖励调权或模型选择。Qwen3-ASR 在非语音音频上的失败可揭示骨干边界，不能用更换任务/删掉音乐题来掩盖。

### 3.10 VoiceBench：生成能力的独立参照

这是不同任务组成的集合：QA、IFEval、开放 judge、安全相关任务、口音/噪声鲁棒性等，分别沿原 evaluator。
只有原始有限答案且不改变证据供给的子任务适合 Choice；其他任务仍需生成回复。
不得先提供 gold 附近候选，再与开放生成原始分数比较；新增 Choice 投影必须有独立名称、干扰项规则和分母。
固定 judge 模型快照、提示、采样参数和失败策略，公布分任务成绩；当前 toolkit 新增任务不能拼入旧论文平均。
[论文](https://arxiv.org/html/2410.17196v1)、[固定 toolkit](https://github.com/MatthewCYM/VoiceBench/tree/b56154172f2a57a43d29005de7d0471d748d70d1)。
不把 judge 均分、指令遵循率或口音问答正确率称为工具成功率。

### 3.11 FDB-v3、τ² voice-v2、EVA：真正执行层

**FDB-v3**：100 条真人语音、mock tools，适合验证自我纠正、多步调用和实时行为；它是固定录音，不等于动态真人对话。
主报官方 Pass@1；tool F1、参数评分、turn-taking、端到端 latency 分列。部分参数/回复由 judge 评分，记录与可执行规则的区别。
[论文 Table 2](https://arxiv.org/html/2604.04847v1) cascade **0.450**、realtime **0.600** 是作者报告；不是本地基线，也不是授权执行 cascade。
新增等待/打断/提前行动标签需时间与状态证据，不能从最终 call 倒推完整因果时机。

**τ² voice-v2**：沿固定 commit `033bf082f48e4a3c6a1256208e93cf2ce9c3bb6f`，六语言×三域×50 任务，900 次运行/trial。
单 cell pilot 只报该 cell；历史论文 278 tasks 与此版本独立。
任务成功需 actor 真实调用、环境状态变更、完整判定与轨迹；oracle DB 重放仅证明桥接组件。
私有用户目标/gold 仅 simulator/evaluator 可见，固定 provider/voices/控制条件与成本。
同版 text-v1 可作外部环境参照，不替代用户要求的直接音频 actor。[环境桥](../../experiments/tau2-environment-bridge.md)。

**EVA-Bench**：clean 213 scenarios/3 domains/k=5；扰动 90/k=3。
EVA-A 与 EVA-X 分列；pass@1 为单次成功，pass@k 为至少一次成功，pass^k 为全部 k 次成功，不能互换。
本地 HF airline 入口不能代表三域全套；provider/judge 未配置时只报资产/接口状态。
[论文](https://arxiv.org/html/2605.13841v1)、[官方框架](https://github.com/ServiceNow/eva/tree/8e101906119e1ddf3314d49913009f3a04327f0d)。

三个交互集共同补报错误执行、重复副作用、撤销/更正、轮数、工具/API 成本、响应延迟 p50/p95；分清离线录音处理与实时等待的计时边界。
只有静态标注时，不自动得到这些指标的 gold，也不据此声称闭环有效。

### 3.12 未纳入主矩阵的其他候选

MASSIVE/ATIS/SNIPS 文本版只供本体和文本学术参考；新增音频需记录真人/TTS 来源、声音模型、许可及文本隔离，不能冒充原生语音基准。
VoiceAgentBench、From Text to Voice 保留为工具候选；VGBench/VoxGate 和 Heard Enough to Act 为 addressedness/行动时机研究线索；MCR-BENCH、MELD、ContextASR/Earnings/ConEC、HeySQuAD 等为证据冲突、情感、实体与 QA 扩展。
发布完整性、监督适配和 scorer 未逐项达到本指南 16 项资源的锁定程度，不列必跑矩阵，不为凑规模扩充下载。

单问、多问、最终判决与显式推理的学习策略另见 [配套分析](09-query-decision-learning-guide.md)。其推荐为原子决策监督、决策包评测，逐步验证联合输出和结构化中间状态；用户最新要求进一步明确：所有 Jev 模型接收可配置思考要求，通过触发与判决增强能力，不采用文本推理路线，见[当前契约](10-configurable-thinking-decision-contract.md)。

## 4. 统一 Jev 数据契约

遵循 [NGA 借鉴方案](../../proposal/04-speech-jev-dataset-v2.md) 的来源闭包、定义重算和配对验证；不照搬其文本截断/70:30 参数。
每份 `xx-jev` 保留源发布与原生任务，派生数据以新 release/hash 发布。

| 文件/字段 | 必须保存的内容 |
|---|---|
| sources / manifest | source revision、许可、音频 SHA/时长/说话人或会话/文本来源、official split、质量排除、代码 hash |
| requests | 音频引用、允许历史、任务定义、独立候选 registry、输出契约、request hash；不含 gold 转写/值/推理 |
| targets | 按 ID 绑定答案、监督依据、available/unavailable/conflict、各字段可评分 mask；与 actor 隔离 |
| rules | rule_id/family、自然语言定义、可执行谓词/映射版本、适用范围；内部 ID 不当答案暗号 |
| pairs | 同义/排列/符号映射/竞争规则/证据变化的关系及 gold 映射；先判关系成立再扩充 |
| split manifest | 来源连通组、learning/selection/calibration/test、规则族及模板族、已有暴露、跨边界交集检查 |

先按 source ID、同源文本/翻译/TTS、会话、重复音频建立连接组，再划分，再生成变体。
官方 split 有跨来源重复时保留原生表；另建去污染面板或排除训练侧相关组，明确改协议，不悄悄改 test。
SLURP 现有 TRAIN 来源闭包已完成，但规则族划分和最终 release 未完成，`optimization_ready=false` 继续有效。

监督状态要严格分开：命题确认为假→false；确认无实体/调用→空集合；候选外/证据不足且协议明确支持→拒答；缺标/冲突→mask。
不能把后三者统一成 null，也不能把标注错误训练成拒答。
未知原因（候选缺失、声学不足、语义冲突）分开评测；原数据没有相应监督时，不宣称已有该能力的测试集。

动态 schema 需同时覆盖：候选排列、符号重绑定、同音频竞争规则、同规则证据改变。
换名字不等于新业务规则；规则改动后重算 gold。同义改写归同族，整族留出，不随机切模板行。
候选菜单由与 gold 无关的预注册规则产生，候选数量/措辞匹配；先菜单后计算覆盖状态，防止缺项标志泄漏。

## 5. 指标契约：主指标、分母与退化控制

所有指标保存整数计数及分母，比例字段注明 0–1 或百分数。原生、派生 Jev、真实交互三个面板分别报告，不合并成总排行榜。
以下是新协议的工作定义；尚未实现的 scorer 需用人工反例验收，不将指南当作实现证据。

| 面板 | 主指标定义 | 必须补充 |
|---|---|---|
| 单意图 Choice | 正确语义候选数 / 全部可评分录音；原生 joint intent 单列 | macro-F1、每类支持数、格式/候选合法率 |
| 多意图/类型集合 | set EM = 整个集合完全相同的录音比例；micro-F1 汇总 TP/FP/FN | macro-F1、各类 P/R、无实体子集、正例支持、无效输出率 |
| 函数存在性 Noul | active P/R/F1；balanced acc = (TPR+TNR)/2 | 函数集合 EM、全不调用/全调用、按函数与 tier 切片 |
| 组合决策 | 同录音意图和整个类型集合同时正确 / 完整可评分录音 | 子任务分数；完整监督覆盖；不能叫完整 SLU |
| 定义遵循 | 每个关系 pair 两边均正确的比例；按变换类别分列 | 语义身份对齐的一致率、判别对正确率、whole-bundle 全对 |
| 拒答 | 无合适候选时正确拒答 recall；拒答 precision；可答样本错误拒答率 | coverage、选择性风险、各未知原因独立统计 |
| 开放抽取/完整调用 | 原生官方 scorer + 明确命名的严格绑定/重数/顺序指标 | 可评分字段/记录覆盖、格式/类型错误、完整任务 EM |
| 闭环 | 官方完整任务成功/pass@1 | 规则/DB成功与 judge 分列、错误执行/延迟/成本 |

**集合计数**：对完整已知标签集合 G 和预测 P，TP=|P∩G|、FP=|P−G|、FN=|G−P|；micro-F1=2ΣTP/(2ΣTP+ΣFP+ΣFN)。
macro-F1 按预注册类别表计算，零支持类别报告 NA 及数量，不在每次运行按表现改变类别集合；所有系统使用同一集合。
无实体预测正确可计 set EM，但整体 F1 分母为零时报告 NA，不凭空给 1。部分监督只评已知字段，记录-level EM 不可判定时标 NA 并报覆盖。

**失败处理**：请求缺失/重复/未知 ID 使完整评测验收失败；明确 runtime error、非法 JSON、无效候选占可评分全分母，EM 计错，不自动修复。
多标签 F1 中无法解析输出作为无有效正预测，金标正例计 FN；负例无效不能计成功 TN。
因此 F1 必须与全分母 EM、格式率共同报告；不虚构一个 FP 数量来替代格式错误。
报告仅有效输出条件下的成绩时，命名为 conditional 并同时展示覆盖，不能作为主要成绩。
未知监督不算模型错误，但必须保留数量与原因，不能悄悄缩小研究对象。

**配对指标**：排列不变按语义身份对齐；符号重绑定按新定义评分；竞争规则按各自 gold 评分。
只测一致性会奖励常量模型，所以每个不变性实验必须同时测正确率和判别性。
whole-bundle 必须固定所需决策清单和规则数量；168 项、6 项、457 项的全对率不直接跨数据集排名。
同一录音的多项相关决策不当作新增独立语音样本。

**拒答/概率**：coverage=有效作答数/可评分样本数；risk=有效作答中的错误比例，coverage=0 时风险 NA。
声称选择性可靠性须冻结拒答阈值和最小覆盖；不能全拒答获得成功。
只有可比且真实导出的概率才报 NLL/Brier/ECE；生成文本中的自报 confidence 不自动作为校准概率。
候选顺序分布稳定不等于概率校准，二者分开。

每个面板至少包含 majority/first-candidate、全空/全不调用等适用退化基线；另做 question-only/零波形/错配音频诊断。
错配音频的“正确答案应变化”需真实监督支持；不能随意换音频仍沿旧 gold，或把波形敏感当理解。

## 6. 横向比较、统计与预算

每条结果键至少为：dataset release + split + metric/scorer hash + input modality + candidate protocol + model revision + training exposure + code/config + seed + inference settings。
任一关键字段不同，分列说明，不直接解释为算法增益。

比较顺序：原生官方参考 → 同骨干同任务的生成式/typed SFT → 同初始化与预算的 SFT/RL/关系奖励消融 → 外部迁移。
开放值若不属主任务，两臂都只做同一决策任务；如比较完整 SLU，两臂都必须有同条件抽取模块，或者明确报告整体系统差异。
不把模型换代、去除坏样本、更多 rollout 或更强教师归因为单个 Jev 设计的收益。

推荐训练 seeds 17/29/43、配对来源聚类 bootstrap 2,000 次/95% CI（继承既有 proposed 统计方案，正式运行配置另冻结）。
SLURP 按来源连接组并披露说话人相关性；FSC 优先说话人；SFC/SpokenWOZ 按会话；Audio2Tool 按源 query；音频 MCQ 按可识别音源组；交互按任务并保留重复 trial。
报各 seed 和 sample SD；来源 bootstrap 不能替代训练随机性。多主检验采用预注册校正，探索切片另标。
若要声称“稳定性改善且准确率不下降”，需预先锁非劣阈值；未锁时只报告二者变化。

训练采样/奖励权重只由 learning 确定；selection 选模型/提示，calibration 校准阈值，TEST 不参与任何反馈。
学习集驱动的离线教师允许分析错误、生成规则草案，但不能成为唯一 gold 来源；actor 始终直接听音频。
SFC 已有不平衡审计可指导其独立训练轨道；Audio2Tool TEST 退化分析仅用于解释指标，不能反向为本轮训练挑权重。

预算分别登记有效来源数、录音小时、决策数、音频前向次数、更新/rollout 数、输入输出 token、设备小时、峰值内存、教师调用/费用。
声明样本匹配或资源匹配中的哪一种，不用“epoch 相同”代替同预算。共享音频编码带来的收益必须实测，端到端计入前处理、全部问题读出和生成。

## 7. 当前证据、发布门槛与下一阶段

| 事项 | 已有证据 | 尚不能声称 |
|---|---|---|
| FSC 原生/派生 | 官方公开权重完整复测；冻结 Omni 父提示/p1 完整独立验收 | 完整 Jev 已可靠、ASR 训练已成功 |
| SLURP | 原始 TEST 诊断；clean-v7 来源划分；48 个来源/142 个时间角色判断的局部审核发布与直接音频 SFT；训练人口诊断 | 全量 clean 训练 release 获准、独立泛化或决策链收益已证明 |
| SFC | 原生/派生数据及 parser/奖励退化审计、学习/选择划分 | 已完成新 ASR 迁移或完整学术重训 |
| Audio2Tool | 输入/目标/派生与完整 scorer 控制验收 | 控制分数是模型成绩、参数缺标已解决 |
| MMAR | 完整输入与 publisher 控制；文献/权重锁定 | 官方 AudioJev 本地完整复测成功 |
| 其余资源 | 固定学术参考与资产记录，部分子任务尚缺访问/适配 | 文件存在即全量可训练、所有基线已复现 |

建议下一阶段按依赖推进：

1. **SLURP 扩展已审核的局部发布**：时间角色先导已完成；后续明确意图/类型定义，隔离冲突与不完备监督，冻结规则族/模板族及候选菜单协议；沿现有来源划分扩展 release，并独立核对 source→request→target→metric。48 个来源的参考文本语义审核不等于全量或人工听审通过。
2. **先证明决策基座的语义能力**：Qwen3-ASR 直接音频先导 SFT 已完成，但训练人口 80/142 判断正确、完整三问仅 5/46 全对，后者与静音相同；下一步为定义区分与独立来源验证。比较同任务生成输出与后续 typed 读出，typed 目前是待实现/验收的研究臂。原始 ASR-only 不遵循任务的失败保留，不用串联补救。
3. **泛化先抓两项**：MAC-SLU 的中文多意图、SFC 的动态函数 ID/OOD；先锁 SLURP-only transfer，再决定是否另设各自 TRAIN 适配。MINDS/SLUE 用于跨语言/自然语音补充。
4. **RL 先消除退化再讨论收益**：全空、全拒、固定候选和关系配对控制通过后，做同预算比较；不能只优化 inactive 平均分。NPU verl/vLLM 链路仍需独立工程验证。
5. **外部与执行验证**：Audio2Tool、MMAR 固定只测不训协议；MMSU/MMAU 评能力边界；最后接一个 τ² voice cell，并用 FDB 真人语音验证。EVA/VoiceBench 不阻塞主线。

每个新 release 必须有数据/代码/scorer hash、许可、来源隔离审计、未知/冲突覆盖、人工反例、完整分母和曝光登记。
结果表固定列：方法、输入、训练来源/预算、数据版本/split、指标、覆盖、seed/CI、成本、证据层级（作者报告/本地复测/新训练/控制）。
最新局部证据见[SLURP 先导](../engineering/slurp-temporal-pilot-v1.md)、[数据与决策链协同](11-decision-chain-dataset-codesign.md)；NPU 当前策略[概率审计](../../experiments/slurp-temporal-rl-parity-v1.md)三个臂均未通过既定阈值，尚未准入 RL 更新。此局部进展不替代全量学术基线复现。
本指南不改变历史发布、不启动新训练、不自动重试任务，也不要求新增模型下载。

用户后续需要确定的研究取舍只有两项：是否正式将首轮主张限定在“意图+槽位类型/函数适用性”；是否将 MAC-SLU/SFC 保持为纯迁移，还是另设监督适配轨道。
在此之前可以继续做无争议的 SLURP 定义审查、数据契约、来源隔离和 scorer 验收，避免等待扩展集全部就绪。
