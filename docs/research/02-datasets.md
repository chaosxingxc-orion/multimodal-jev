# 数据集、任务与评测

选择原则：先覆盖语音语义决策，再覆盖动态函数定义和执行；每份资源区分原始人声、朗读、TTS 与交互模拟。表中许可只代表所查发布页面，底层来源条款仍需随下载保留。[来源索引](05-sources.md)

## 1. SLU 主线

| 资源 | 数据/任务 | 适用性与划分 | 许可及限制 |
| --- | --- | --- | --- |
| **SLURP** | 英语语音，意图+实体/槽位；18 场景、训练清单 91 个意图 | 第一主基准。官方 real train/devel/test 分别 **50,628 / 8,690 / 13,078 录音**；文本记录数分别 11,514 / 2,033 / 2,974，同一文本可有多个录音 | 文本 CC-BY-4.0；音频 CC-BY-NC-4.0。synthetic 训练录音单列，不混入 real 计数 |
| **MAC-SLU** | 中文车载多意图，8 域、81 意图、192 槽位，最多 5 个意图；真实命令文本+TTS 音频 | 检验多意图、槽位归属、中文迁移；按原始 train/dev/test | HF 卡标 Apache-2.0，但 language 元数据写 en，与论文中文描述不一致，以实际记录核验语言 |
| **MINDS-14** | 多语言银行场景意图，14 类，含音频/文本 | 较小的跨语言诊断；HF 主要是 train split，需预注册自有 train/dev/test，不能把随机划分称官方测试 | CC-BY-4.0；聚合 `all/` 与各语言分片避免重复下载 |
| **Fluent Speech Commands** | 30,043 条、97 说话人、31 个 action/object/location 组合意图 | 适合管线 smoke 和低成本基线；说话人划分，不等于词句组合泛化 | 官方 Fluent Speech Commands Public License 限定学术研究；不能仅以社区 CC 标签代替该专用条款 |
| **SLUE Phase 1** | 自然语音 NER（VoxPopuli）、情感（VoxCeleb） | 比朗读命令更自然；NER micro-F1、label-F1；情感 macro recall/F1 | VoxPopuli 与 VoxCeleb 子集分别带许可；toolkit MIT 不代表数据 MIT |
| **SLUE Phase 2** | 语音 QA、摘要、实体时间定位、对话行为 | 优先实体定位/对话行为，QA 作证据选择扩展；不同任务不能聚成一个准确率 | 按子集许可；原始 zip 与 Parquet 是不同发布视图，全部拉取会重复占空间 |
| **SpokenWOZ** | 5,700 真实人对话、约 203k turns、249 h、8 域 | 跨轮槽位与对话状态；joint goal accuracy；官网已提供 train/dev 及 fixed test，旧 README 的 hidden test 说明已过时 | CC-BY-NC-4.0；静态对话回放不是新的交互任务成功实验 |
| **MASSIVE / 常见 ATIS、SNIPS 文本版** | 意图/槽位文本资源 | 可供 schema 和文本对照；不能仅凭“SLU”命名当作有音频的数据 | MASSIVE 是 SLURP 相关文本本地化；需按具体语音扩展另查来源 |

SLURP 的计数本次直接解析官方 GitHub 三个 JSONL，按 `recordings` 求和；保存在 [核验记录](dataset-audit.json)。官方 test 只出现 77 个意图，devel 71 个，不因此删除训练定义中的其他意图，也不把 scenario×action 的笛卡尔积当标签集合。不同录音、麦克风和同一句话要保留对应关系，配对统计应以语句/说话人聚类而非假定所有录音独立。

## 2. 从 SLU 到语音函数调用

| 资源 | 核心覆盖 | 发布与评分 | 本课题优先级 |
| --- | --- | --- | --- |
| **SFC-Bench / SpokenFC** | 动态函数定义、缺参数 `NAN`、单/多意图、单/多轮；300 函数，232 ID / 68 OOD | 官方仓库：处理后 SFC train-ID 5,534、test-ID 2,590、test-OOD 771；函数名、参数值 P/R/F1、整体 exact accuracy；CC-BY-NC-4.0 原创材料 | **高**。直接测试“读定义”及未见函数；TTS 合成，需真人外部验证 |
| **Audio2Tool** | 直接、带参、多意图、隐式、干扰、自我纠正、多轮、前景/背景混合 8 tiers；152 工具 | HF 卡 16,843 queries、36,421 文件、59.6h；CC-BY-NC-4.0；当前 configs 均为 test | **高**。适合类型化动作与参数；禁止拿 test 随手微调 |
| **VoiceAgentBench** | 单工具、工具检索、并行/顺序调用、多轮、拒绝不当请求 | 多语言合成语音；expected tool calls；自定义 Krutrim 社区许可 | 中。先核实发布许可、scorer、全部分片完整性 |
| **From Text to Voice** | 将 Confetti / When2Call 转成带声音变化和噪声的配对语音任务 | 保留文本 schema 和 gold；结构匹配是诊断，不是完整环境执行 | 中。适合控制文本→语音变化、何时调用 |
| **SFC/MAC-SLU 与 Audio2Tool 的组合** | 前者提供训练与 OOD 函数；后者提供独立复杂声音条件 | 对 schema、工具名、原始文本和音频来源做跨集重复检查 | 优先组合，不能先看测试再修标注或阈值 |

Audio2Tool 论文约 30k 的口径与发布卡不一致，不能挑较大的数做实验规模。卡片 quickstart 用 `split="train"` 加载 JSON 是加载器名称，不赋予其训练用途。`query`、`expected_tool_call`、`extracted_params`、`reasoning` 等 gold 字段必须与模型可见字段分开；“给出的 functions 是否只是正确函数”也应通过实际记录核查，避免候选泄漏。

SFC 在论文与代码中明确存在运行配置差异：仓库 GRPO 默认 BF16/8 generations，论文最终配置为 FP16/32 generations。复现结果须对应自己的实际版本，不能仅凭执行默认脚本就称复现论文。

## 3. Agentic Speech：执行、时序与完整任务

| 基准 | 音频来源/规模 | 真正测什么 | 接入与局限 |
| --- | --- | --- | --- |
| **FDB-v3** | **真实人声**和自然不流畅表达；发布 README：100 examples、79 scenarios、12 speakers、4 域、12 mock 工具；48kHz 输入 | 多步调用、参数、pass/fail、时延、turn-taking；self-correction 难例 | 高价值真实语音验证；LiveKit/实时前端适配；样本量小，不能单独证明宽域泛化；数据发布在 Google Drive，具体数据条款需归档 |
| **τ-Voice / τ²-bench voice** | 合成用户语音、噪声/口音/打断；论文 278 tasks | 环境中的可验证任务完成与语音交互质量 | 当前 main 已有 voice-v2、多语言协议，与论文初版不同；必须固定代码与场景/声音版本；默认 provider/声音依赖外部服务，下载代码并不等于完全离线可运行 |
| **EVA-Bench** | bot-to-bot 合成场景，论文 213 scenarios、3 企业域 | EVA-A 完成/准确、EVA-X 体验；多次试验可靠性和噪声/口音扰动 | GitHub 框架已发布；所查 HF 卡仅 airline 入口，不能把它等同三域完整套件；部分指标依赖 LLM judge，需与可执行规则分列 |
| **SpokenWOZ** | 真实人声多轮对话 | 已标注状态追踪、静态任务语义 | 可作为接入前的状态层测试，但不是可与 τ-Voice 直接排名的交互基准 |
| **VGBench / VoxGate** | 1,018 项 addressedness 诊断，side-talk/self-talk/speaker-switch | 该不该动作：沉默、工具调用、口头回应 | 是声学与上下文联合干预，不能把结果都解释为纯声纹识别；发布完整性仍待核验 |
| **Heard Enough to Act** | 合成 26,694 对话、786h，48 域（作者报告） | earliest actionable point，因果分块时机与参数准备 | 论文说将发布，未据此认定全部数据已开放；我们不能只换成 Jev 头就声称首次研究提前行动 |

`pass@k`（至少成功一次）与 `pass^k`（所有 k 次成功）不可互换。提前执行要同时测错执行、撤销、更正、重复副作用和完成成本；提前生成文本并不等于安全完成工具调用。

## 4. 证明语音证据价值的诊断集

| 资源 | 作用 | 注意事项 |
| --- | --- | --- |
| **MMSU** | 5,000 audio-question-answer、47 任务，语义、音韵、韵律、副语言 | 与 AudioJev 的 3,931 短音频子集不同；默认完整评测，时长限制须明示 |
| **MMAU** | 语音/环境声/音乐多选问答；完整 10k 与 mini 1k 不同 | 已分别选择 1,000 条 mini（公开答案）和 9,000 条 test（隐藏答案）；音源跨集重复检查 |
| **MMAR** | 1,000 高难混合音频推理 | 作为外部诊断，不以它替代 SLU/工具成功 |
| **MCR-BENCH** | 音频与文本冲突 | 检查模型是否只相信文字提示、忽略声音 |
| **MELD** | 对话情感、韵律/语义控制 | 常见角色跨 train/test；不能直接声称说话人泛化；代码许可不能代替原始视频音频许可 |
| **VoiceBench** | 自然语音和 TTS 混合，指令、知识、鲁棒性等 | 泛用助手能力诊断；其 judge 分数不是工具执行成功率 |
| **ContextASR-Bench / Earnings / ConEC** | 实体密集和上下文偏置 ASR | WER、实体错误与下游错误分开；正确实体清单可能泄漏答案 |
| **HeySQuAD / SLUE-SQA-5** | 语音问答、证据定位 | 判断音频究竟是问题还是文档；提供完整 gold passage 的任务与只给检索文本不同 |

## 5. 数据入库和实验规则

1. 使用 `datasets/<资源名>/` 作为共享入口，保留 revision、文件清单、许可、下载日志与校验状态；缓存和解压产物区分。
2. 先核验已有 SLURP / Audio2Tool，避免重新拉取相同音频。资源“目录存在”与“所有文件完整可读”分别记录。
3. 首批重点包括 SLURP、SFC-Bench、MAC-SLU、MINDS-14、Audio2Tool、SLUE、MMSU、MMAU、MMAR、VoiceBench，以及 FDB/τ/EVA 的发布资产。依赖动态服务的音频不能伪装成已下载的固定数据集。
4. 未发布、受访问门槛限制、条款不明或链接失效的资源记录具体原因，不用社区同名包替换官方发布。
5. 先固定训练、校准、测试及工具/说话人/模板分组，再生成扩充样本。语义不变与语义改变成对构造，使用人工抽查约束 TTS/变速/噪声是否改动标签。

2026-11-09 本地核验：Audio2Tool 固定 revision 实际为 36,419 个 WAV、71,441 个总文件，与卡片 36,421 个音频的口径有差异；不以卡片数字替代实际清单。
