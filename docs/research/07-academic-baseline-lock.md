# 学术基线锁定 v1

调研日期：2026-11-10（Asia/Shanghai；来源访问时间保留 UTC）。本次锁定的是**比较对象、证据口径和复现路径**，调研子进程没有运行训练、模型评测或设备验证；主进程同期运行验证在下文另引工程凭证。不是穷尽最新 SOTA 的排行榜。机器登记见 [academic-baseline-lock.json](academic-baseline-lock.json)，来源和失败记录见 [academic-baseline-sources-v1.json](academic-baseline-sources-v1.json)，执行门槛见 [横比方案](../../experiments/academic-baseline-comparison-v1.md)。

## 1. 锁定结论与边界

首轮主表采用 FSC、SLURP 的原生任务；SFC 与 Audio2Tool 各建外部函数调用表；MMAR 单独建 AudioJev 原顺序复测表。优先复测已有公开权重的官方参考，随后开展同骨干受控重跑。不能用自己的 frozen Qwen head 代替学术基线，也不能因缺少外部 checkpoint 就给它填一个推测分数。

| 数据集 | 必需原始/官方参考 | 适合本研究的强对照 | 本次锁定的复现性质 |
| --- | --- | --- | --- |
| FSC | Lugosch 2019 预训练后冻结声学模块、训练 intent RNN | SpeechBrain LibriSpeech ASR 初始化的 direct seq2seq | 前者重训复现；后者官方权重复测，权重 revision 已查 |
| SLURP | 原论文 Multi-ASR→HerMiT，官方 scorer | SpeechBrain HuBERT direct；ESPnet deliberation 作转写辅助 intent 外部对照 | HerMiT 文献参考；SpeechBrain/ESPnet 有官方权重。real-only 改配方须单独重训 |
| SFC-Bench/SpokenFC | 官方 Qwen2.5-Omni-7B prompting 与 SFT | SpokenFC-7B fine-grained GRPO，配 EM-reward 消融 | 基座可复测；任务权重未核验，SFT/GRPO 只能列文献/重训轨道 |
| Audio2Tool | 原论文 Qwen2.5-Omni-7B zero-shot | Whisper v3→Qwen3-8B，配 gold query→同文本模型 | 官方代码已固定，但当前 scorer 与论文指标有差异；历史运行配方待核 |
| MMAR | 原 MMAR 官方 Qwen2-Audio-Instruct + publisher scorer | AudioJev 官方单 checkpoint，并列 Frozen Omni 3B 同骨干参考 | AudioJev 权重、代码、输入协议已固定；CUDA 推理未完成 |
| τ² voice（后续） | 固定 commit 的 voice-v2 官方环境/评测 | 同版 text-v1、原生 realtime provider，再接本项目决策节点 | 当前仅 oracle 环境验证；没有模型/语音端到端分数 |

“原始/官方可复现参考”在此区分**可识别官方方法**与**执行已经就绪**：Audio2Tool 当前官方 scorer 与论文协议尚未对齐、SFC 缺任务 checkpoint 和完整 SFT 配方，不能声称这两项满足即刻复现。登记中用 blockers 明确保留缺口。

比较分三层，层间不求总排名：

1. **L1 原生官方任务**：原始标签、划分、输入、scorer 和训练条件；原生 FSC joint intent、SLURP SLU-F1、SFC 调用输出、MMAR MCQ 各自建表。
2. **L2 派生决策任务**：schema permutation、定义 rebinding、missing-candidate、Noul/缺失状态；新增指标与原任务分列。原文论文分数不能直接迁入此层。
3. **L3 真实交互执行**：模型决定调用、接收真实环境响应、维护状态，再以任务完成和交互质量计分。SFC supplied history、SpokenWOZ 静态回放、τ oracle DB replay 都不能替代这一层。

## 2. 核心数据集与精确参考

### 2.1 FSC：保留原生 joint intent 主表

原始论文 [Lugosch et al., 1904.03670v1](https://arxiv.org/pdf/1904.03670v1) Table 1 给出 train/valid/test **23,132 / 3,118 / 3,793**，Table 2 明确为 test。正确标准是 action、object、location 三者全对；不是把三个槽位平均，也不是本项目六视图平均。

| 原生 test 参考 | 论文/发布定位 | 作者报告 | 训练与输入条件 |
| --- | --- | ---: | --- |
| No pre-training | v1 Table 2，Accuracy (full) | 96.6% | 全量 FSC，随机声学初始化；退化参考 |
| No unfreezing | v1 Table 2，Accuracy (full) | **98.8%** | LibriSpeech 音素/词预训练、冻结预训练声学模块，训练语义模块 |
| SpeechBrain direct | 官方 [recipe Results](https://github.com/speechbrain/speechbrain/tree/c0241278f93ef02cd5191f03caf4e5f8e1292bf6/recipes/fluent-speech-commands)，`21-06-03 / train.yaml / Test Acc` | **99.60%** | LibriSpeech seq2seq ASR 初始化、FSC supervised direct 训练；外部预训练与增强须披露 |

原实现 [end-to-end-SLU](https://github.com/lorenlugosch/end-to-end-SLU/tree/e94bd479de1a82663c38363979309089acebcd36) 已核 README 和 `experiments/no_unfreezing.cfg`，其中 `unfreezing_type=0`、全量 real、synthetic speaker 比例为零；README 声明提供 LibriSpeech 预训练模型，本次未读取 checkpoint。公开 [SpeechBrain 权重](https://huggingface.co/speechbrain/slu-direct-fluent-speech-commands-librispeech-asr/tree/9163a11ee29a14241356a7cc2cc05672b73ba5fe) 已核 HF revision 与文件元数据。主进程随后下载并校验该权重及独立 LibriSpeech ASR 依赖，官方 `decode_batch` 已完成一条 TRAIN 音频推理；见 [资产记录](../engineering/fsc-speechbrain-assets-v1.json) 和 [探针结果](../engineering/fsc-speechbrain-probe-v1-result.json)，后续主进程已完成完整 TEST3793 的公开权重复测：joint 3774/3793（99.4991%），序列精确匹配同值，见 [独立复核结果](../engineering/fsc-speechbrain-official-v1-test-result.json)。这比作者报告99.60%低0.1009个百分点，属于当前固定官方推理/CPU环境的任务训练权重复测，不是零训练基座，也不是历史训练全过程复现。原学术调研与后续实测证据分别保留。当前 recipe commit 不是历史训练 commit，不能由两者同时可访问推断逐参数复现。后续已解析模型卡短 hash 为 `f1f421b3bb58dabc75d67c3fd5f6e3359943b927` 并读取历史配方；默认值不证明实际训练覆盖项，历史脚本的 WER/CER 也不能替代 accuracy，详见 [历史源码追溯](speechbrain-history-v1.md)。当前 recipe 的 beam=80、temperature=1.25 只作为已读配置，权重复测须以随权重 hyperparams 为准。FSC 数据仍受原发布方专用学术许可约束，模型卡的 CC0 标签不覆盖语料许可。

FSC 已完整下载的状态继承本地验收。本项目 **94.1522% ± 0.2275 pp** 是 [frozen head valid 三 seed 结果](../../experiments/fsc-frozen-audio-full.md)，不是 test，也不是上述学术方法复现。现有 **31 条训练录音 / 186 视图**的 SFT/RL 是开发诊断；六视图来自同源录音，不能算独立扩大的样本量。后续要同时交付全量 native test 和独立的六视图 test 表。

冻结Qwen3-Omni父提示后续已完成完整官方TEST的3793录音×6视图并独立复核：
native严格JSON候选选择1752/3793（46.1904%），六视图全对9/3793（0.2373%），
见[完整结果表](../../experiments/fsc-omni-official-test-v2-parent.md)。这是无FSC权重
训练、带严格消费格式的端到端基线，与上述监督训练分类器的条件不同；不能
将它称为学术模型复现。已冻结p1的同协议TEST对照仍在运行，尚无完整提升结论。

### 2.2 SLURP：实体指标、intent 和训练来源分别锁定

[EMNLP 2020 原论文](https://aclanthology.org/2020.emnlp-main.588.pdf) Table 6：real train/dev/test **50,628 / 8,690 / 13,078 录音**；synthetic **69,253** 来自已有本地完整性验收，另列，绝不并入 real。文本条数与录音条数不同，多麦克风/同文本录音应保留来源关联。

| 参考 | 定位 | 作者报告 | 纳入位置 |
| --- | --- | --- | --- |
| Multi-SLURP/HerMiT | 原论文 Table 8 / 同名行 | SLU-F1 **70.84**；Word-F1 69.34；Char-F1 72.39；实体 exact F1 64.16 | 外部 ASR 训练/适配的历史原生参考 |
| 同一 HerMiT 管线 | Table 7 / 同名行，Scen_Act | **78.33%** | intent 列，不能当 SLU-F1 |
| SpeechBrain Direct (HuBert) | [模型卡](https://huggingface.co/speechbrain/SLU-direct-SLURP-hubert-enc/tree/6644fb49e209d2c4c3ea6bfd8accbe06b2df28c4)，`30-11-21`；recipe Performance summary | **75.10 SLU-F1**；recipe 另报 **87.54% intent** | 官方权重参考；当前配方为 real+synth |
| ESPnet 2-pass deliberation + ASR transcript | [官方 README](https://github.com/espnet/espnet/blob/master/egs2/slurp/slu1/README.md)，`valid.acc.ave_10best/test` | **86.8% intent** | 音频+ASR 转写辅助，单独表 |
| ESPnet 2-pass deliberation + gold transcript | 同 README 的 ground truth 条件、同 test 行 | **89.0% intent** | 音频+人工文本证据参考，不能冒充纯音频 |

SpeechBrain 的 [固定当前配方](https://github.com/speechbrain/speechbrain/blob/c0241278f93ef02cd5191f03caf4e5f8e1292bf6/recipes/SLURP/direct/hparams/train_with_wav2vec2.yaml) 使用 `facebook/hubert-base-ls960`，`train_splits=[train_synthetic, train_real]`，冻结卷积部分、其余 encoder 可训练，attention decoder；beam size 80、temperature 1.25。模型卡声明历史训练 commit `aa018540`，初次按当前路径查询返回 404。后续已解析为 `aa0185408025e80f6c748d2c7af7fa96958c2231`：完整 tree 无卡片引用的 wav2vec2 训练文件，历史 direct 配方是 LibriSpeech ASR 编码器而非 HuBERT，见 [源码追溯](speechbrain-history-v1.md)。故**历史 commit 可解析仍不证明 HuBERT 权重的训练配方已核实**。real-only 强对照采用同 recipe 显式去掉 synthetic 重训，分数空置。

ESPnet 的两个官方 HF 权重 revision 已登记；README 声称的历史 git hash 尚未独立核验，登记 `code_revision=null`。特别避免沿用 [ESPnet-SLU 2111.14706v1](https://arxiv.org/pdf/2111.14706v1) Table 5 中混杂的 `IC (F1)` 列来替换 SLU-F1：该表把 SpeechBrain 75.1 放在 IC 列，和官方模型卡含义不同。本报告优先采用明确原始指标来源，保留论文作方法参考。

**官方评分实现已复核**：[SLURP commit 8eb1654](https://github.com/pswietojanski/slurp/tree/8eb16545762be97ace75334109d73824217311f1/scripts/evaluation) 的 `evaluate.py`、`util.py`、`metrics/metrics.py`、`metrics/distance.py`。`SLUF1` 合并 word/char 的混淆计数；不是简单拿两个 F1 求平均。`--load-gold` 以文本 `slurp_id` 评分，普通音频预测以录音 filename 评分；二者分母不等。`load_predictions` 用字典写入，重复 ID 会覆盖；`evaluate.py` 对缺失预测只警告且不进入该例计数。正式复测须在官方 scorer 前独立要求 ID 一一对应、全覆盖、拒绝重复，不能以少输出获得好分数。scorer 本次为代码静态审阅，尚未运行新 fixtures。

[已有 SLURP transfer 协议](../../experiments/slurp-transfer-v1.md) 留出业务域并生成不同任务，不能把其选取录音数、accuracy 或 null 指标与官方全 test SLU-F1 混排。本报告不修订它。

### 2.3 SFC-Bench/SpokenFC：训练轨道与外部 transfer 分开

锁定 [2608.05126v2](https://arxiv.org/html/2608.05126v2)，官方 [FunResearch commit 5cd9dc6](https://github.com/QwenAudio/FunResearch/tree/5cd9dc679584ba9b14454908a0f2c85e74c00147/SpokenFC)。Table 1 是 conversation **4,499 / 1,986 / 593**；处理后 train_id/test_id/test_ood 是 turn **5,534 / 2,590 / 771**。官方两个计数层级一致，不能以展开 turn 数解释为新增独立会话。

| Table 7 的方法行 | test-ID Overall | test-OOD Overall | 条件 |
| --- | ---: | ---: | --- |
| SFT | 71.90 | 73.93 | Qwen2.5-Omni-7B LoRA SFT |
| RL w/ EM Reward | 73.08 | 73.24 | 精确匹配奖励对照 |
| SpokenFC-7B (Proposed) | **74.15** | **75.10** | fine-grained GRPO、FP16、N=32 |

这些是作者报告的 turn-level official accuracy，不是我们实测，也不是工具执行成功。Table 5 `LALM / Qwen2.5-Omni / Average (Turn-Level)` 为 **49.03**；Table 7 基座行为 **48.80**，后者数值又对应 Table 5 的 ASR+LLM 同名行。表间路由/设置不足以确定完全一致，登记固定精确行，不拼成“49.03→74.15”的受控因果收益。

论文 §5.3 Post-Training 描述只对 LLM backbone 做 LoRA，冻结 audio/vision/talker，训练一个 epoch；原硬件为 NVIDIA A800。仓库明确 SFT 使用 LLaMA-Factory，未在此完整复制；GRPO 默认 BF16 / 8 generations 与论文最终 FP16 / 32 不同。**本次未找到已核验 SpokenFC task checkpoint**；基座 Omni 权重不能充当论文 SFT/GRPO 权重。将来自己重训需要独立预算和配方，不承诺立即复现。

沿用 [现存 SFC external protocol](../../experiments/sfc-external-evaluation.md)：当前 Jev 仅从 FSC/SLURP 训练，SFC train 为参考，不混入训练。若要比较 SFC 内域重训，是另一个 proposed 实验，必须另立协议。所有级别、多轮两段音频、官方 supplied assistant clarification、原始 `NAN`、数组与调用重数全部保留。论文称 speaker isolation，但发布记录不提供 speaker ID，本项目只可声明已验音频哈希/函数集合隔离，不能自行认证说话人隔离。

官方 scorer 与 strict scorer 并列，保留原始 prediction 和官方 stdout。已知官方 parser 对数组、跨函数参数归属、重复调用等有问题，证据见既有 [scorer audit](../engineering/sfc-scorer-audit.json)。strict 补充列用 call-multiset exact、function-bound slot F1、两轮 both-correct；不能静默把 strict 替换为官方数。现有 two-audio CPU engine check 输出是转写，只证明请求路径接受，未证明任务成功或两段音频因果使用。

### 2.4 Audio2Tool：最相关的强管线对照，但官方评分实现仍是门槛

后续 loader 审计还确认候选和上下文协议问题：release loader 缺少入口要求的
`set_tools_schema`；备用 schema 从 gold 主工具/参数键构造，遗漏仅附加调用工具。
旧 Tier7 loader 的轮次列表被通用 evaluator 当 speaker 变体逐条评分。人工反例见
[官方代码审计](audio2tool-official-code-audit-v1.md)。正式接入必须使用独立固定 registry、
明确保序多轮输入与全分母，不能直接把默认入口作为完整论文复现。

锁定 [2604.22821v1](https://arxiv.org/html/2604.22821v1) §5.1–5.2、Table 3，[官方页面](https://audio2tool.github.io/) 与 [HF 发布](https://huggingface.co/datasets/RVtech/Audio2Tool/tree/f1388da9a3189541ab82adac88824a0661670c43)。论文约 30k 与卡片 16,843 queries 不同；卡片 36,421 音频、本地固定 manifest **36,419 WAV / 71,441 总文件**亦有差异。后者继承 [dataset-audit](dataset-audit.json) 与本地资产验收，本次未重扫音频。

官方仅 test。query、expected_tool_call、extracted_params、reasoning 不进入音频系统输入。候选 schema 供给规则需逐记录审计，防止只给正确工具形成泄漏；TTS 变体、tier、来源 query 需要聚类，不能当独立问题简单加权。

| Table 3 行 | Tier 1 tool Acc | Tier 2 EM | 定位 |
| --- | ---: | ---: | --- |
| Qwen-2.5-Omni-7B | 79.5 | 14.7 | 原生语音 zero-shot 官方参考 |
| whisperv3 + Qwen 8B | 78.1 | 10.8 | ASR→Qwen3-8B 强管线对照 |
| Qwen 8B | 85.6 | 10.1 | 原始 query 文本输入；独立证据对照 |

这些数字不构成跨列排名；它们显示选对工具与参数全对有明显差距，不能用 tool accuracy 宣称完成任务。语音+大模型不是在所有 tier/指标上都占优，强对照的选择依据是研究相关性与可解释性，而非宣称处处最高分。

论文规定多调用 **ordered tool sequence** 匹配，EM 经 deterministic normalization，slot 为 micro-F1；因此不可直接套 SFC 无序集合 scorer。后续已定位官方仓库并固定 commit `b9d5383ca30945fbafb965a04bea40117e1e8a82`，见 [官方代码审计](audio2tool-official-code-audit-v1.md)。实际指标反例表明当前代码使用工具集合匹配、逐样本平均参数 F1；聚合静态审阅还发现错误样本被排除。当前官方代码指标须独立列示，不能冒充论文 ordered/micro 协议；固定 release loader、历史 prompt/checkpoint 与精确配方仍待核验。人工 transcript 不覆盖 Tier 8 的前景/背景证据，因此不是必然性能上界。

### 2.5 MMAR / AudioJev：最优先的官方 typed-decision 对照

采用原 [MMAR](https://arxiv.org/html/2505.13032v1) 1,000 题答案正确率，不用后来的 MMAR-Rubrics CoT judge 取代。原基准 [官方 README](https://github.com/ddlBoJack/MMAR/tree/3bce090a967db576c8ad433b290a8da582d6d2a0) 的 `Qwen2-Audio-Instruct / Avg` **30.0** 为原生生成式参考，权重元数据已固定，但不能证明当前 HF revision 就是作者当时运行版本。

最重要的对照是 [AudioJev 2610.01293v1](https://arxiv.org/html/2610.01293v1)：

| 分数 | 来源/行 | 正确解释 |
| --- | --- | --- |
| **55.33 ± 0.84%** | Table 1 AudioJev/MMAR；Table 9 三 seeds | 论文三 seed mean ± sample SD |
| **56.30%** | Table 9 seed01；[模型卡 Evaluation](https://huggingface.co/shlv/AudioJev/tree/d23dddb9669021bb2ef8cae5c07fe24b3e35702d) | 唯一此次核验公开 checkpoint，seed 20261001，general step 2048，原始顺序 |
| **53.50%** | Table 1 Frozen Omni 3B/MMAR | AudioJev 同骨干未做目标任务拟合的候选 logits 参考，需独立复测适配 |

推理为候选位置 token logits softmax，一次 forward，不是生成 JSON。RD-SKL 使用候选身份对齐后的随机错排双视图，训练不等于经验概率校准。Table 5 的无 paired training MMAR accuracy **55.93 ± 0.42**，高于 RD-SKL 的平均准确率；RD-SKL 的收益主要需要以顺序稳定性联合正确率解释，不能声称已有论文证明准确率必然提高。

代码固定 `8eeeab5e77d2c410fd6058e7bdf2b4125f11903e`，HF 权重 `d23dddb9669021bb2ef8cae5c07fe24b3e35702d`。此次重读 [backend](https://github.com/SihanLv/AudioJev-Inference/blob/8eeeab5e77d2c410fd6058e7bdf2b4125f11903e/audiojev_inference/backend.py) 确认显式 CUDA 检查和 BF16。175 的 NPU 4–7 使用授权不能解决 CUDA 依赖；未经验证的 Ascend 移植必须单列为 port，不叫忠实官方复现。

延用 [MMAR 冻结协议](../../experiments/audiojev-mmar.md)：完整音频、原始候选顺序、重复/空白候选和非单选 reference 保留；使用官方 renderer/backend 绕过服务 wrapper 的 canonical 排序。已有完整 1,000 条输入、publisher scorer/oracle 控制，**没有模型成绩**。不能用 always-first/参考可满足上界代替模型结果，也不因五个 reference 在该单候选输出契约下不满足而删除它们。

## 3. 已计划扩展集：逐项基线与纳表决定

这些资源都登记，不扩大下载范围。除 MAC-SLU 原生任务扩展表外，不进入首轮 FSC/SLURP 主表；每个任务使用自己的指标与证据类型。

| 数据集 | 锁定学术方法与精确来源 | 评测口径、训练条件、复现性质 | 纳表决定/阻断 |
| --- | --- | --- | --- |
| MAC-SLU | [2512.01603v1](https://arxiv.org/html/2512.01603v1) Table 6：Qwen2.5-Omni-3B SFT；Paraformer+Qwen3-8B SFT；Qwen3-8B gold-text SFT | 同表 Overall Acc **56.56 / 47.18 / 60.73**；意图 Acc、slot F1 另列。ASR 不微调、文本 Qwen 做 NLU SFT。Table 3 train/dev/test **17,997 / 1,391 / 1,151** 为文本记录；保留多意图绑定 | 中文 TTS 泛化扩展表。官方 [MAC_SLU](https://github.com/Gatsby-web/MAC_SLU/tree/d54453f63e0dc57b17d3cb2cfea35fb7da6a4489) 有 ICL/scorer，但 SFT 仅通用指导；未核 task weights，不能称权重复测 |
| SLUE Phase 1 | [2111.10367v2](https://arxiv.org/html/2111.10367v2)，[toolkit](https://github.com/asappresearch/slue-toolkit/tree/ed23039507c8f01b704a5c43ed89b6e808a49405) 的 `baselines/ner`、`sentiment`：wav2vec2 E2E，与 ASR→DeBERTa 配对 | NER phrase+tag micro-F1、label-F1；SA macro recall/F1 分开。fine-tune/dev/test 原分区；NER LM/no-LM 与 large/base 不混。VoxCeleb 使用 v0.2 分布修订 | 自然语音扩展表，**gated 音频未齐**。recipe 可重训；部分 README 命令/模型名不一致，需核脚本，不填推测分数 |
| SLUE Phase 2 | [2212.10525v2](https://arxiv.org/html/2212.10525v2)：Table 5 DAC wav2vec2→DeBERTa；Table 6 QA pipeline 与 DUAL；Table 7 SUMM；Table 8 NEL | DAC test macro-F1 **70.7**。QA test frame-F1：pipeline-w2v2 **39.6**、E2E-DUAL **21.8**，verified-test 是另列。SUMM 为 HuBERT-Conformer→LED 及 E2E，ROUGE/METEOR/BERTScore；NEL 为 CTC NER 时间映射或 ASR→DeBERTa，frame-F1/不同 overlap 阈值 word-F1；v2 Table 8 是 **test**（E2E-w2v2 frame-F1 **56.3**），Table 18 才是 dev precision/recall，Table 19 分列 dev/test | 优先 DAC/NEL 扩展；QA/SUMM 后续。数据已下载不等于模型适配。NEL 没有独立 train，复用 Phase1 NER，仍可能受 gated 限制。v1 未给 QA 分数，故此次明确升级文献版本到 v2，不把 v1 表号套用 |
| SpokenWOZ | [2305.13040v2](https://arxiv.org/html/2305.13040v2) Table 7：SPACE+WavLM_align，配 SPACE text-only | JGA **25.65**；音频+文本双模态，全部槽位；不是 `-w/o cross-turn slot` 列。官方 [DAMO-ConvAI/spokenwoz](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/faf754671f5b850e1b4d1ff870db37a6a2480df6/spokenwoz) 有 finetuning 实现 | 状态追踪扩展表；response Inform/Success/BLEU 单列。当前 fixed test 需与论文 test 核对。静态 Success 不能和 τ 真实任务成功混排 |
| MINDS-14 | [2104.08524v1](https://arxiv.org/pdf/2104.08524v1) Table 1：Google ASR→frozen LaBSE→MLP，target-only standard | Average **93.4%**，含 AUX-TARGET；§4 为 60/40 随机划分、文中称 3-fold，并平均三次；非 HF 官方 test。translate-to-EN/no-aux 是不同条件 | 多语言诊断表。HF 主要 train，原 fold IDs 未核验。AudioJev **672 条、六语言变体、88.59 ± 0.23**（Table 1）是另一子集，另栏，不能横排 |
| MMSU | [官方 MMSU](https://github.com/dingdongwang/mmsu/tree/217d1d3f5ad59203ed16f44c0fd913f4f8add2b4) Leaderboard Qwen2.5-Omni-7B Avg(All) **60.57**；[2506.04779v1](https://arxiv.org/html/2506.04779v1) | full **5,000 / 47 tasks**，维度及任务聚合须核 evaluator；AudioJev 卡片 **62.45** 只对应 **3,931 短音频 / 39 families**，不可比较为 full 改进 | 声学证据诊断，full 主列、short 独立复核。长音频截断/样本排除不得静默进行 |
| MMAU | [2410.19168v1](https://arxiv.org/html/2410.19168v1) Table 3 原始 Qwen2-Audio；[官方代码](https://github.com/Sakshi113/MMAU/tree/110127f54c0dfba3faa5ec9feee4a7e4148679c5) | mini **1,000** / hidden **9,000** / full **10,000** 分列。发布已更新 `MMAU-v05.15.25`，旧论文数字不直接搬到当前 release。AudioJev 单权重 full **69.41** 与三 seed **68.88 ± 0.46** 分开；原始顺序 | 音频泛化外部表。HF Space 官方 hidden scoring；作者披露 **119 hidden 音源**与拟合/开发资料重叠，不能称无污染泛化 |
| VoiceBench | [2410.17196v1](https://arxiv.org/html/2410.17196v1) Table 2 Naive=Whisper-large-v3→LLaMA-3.1-8B-Instruct；配同文本骨干 LLaMA-Omni；Table 3 分任务结果 | [toolkit](https://github.com/MatthewCYM/VoiceBench/tree/b56154172f2a57a43d29005de7d0471d748d70d1) 按 open judge/QA/IFEval/harm 等不同 evaluator，sd-qa 按口音；当前 README 的 gpt-4o-mini judge 需与论文设置分列 | 通用助手诊断表，不能把 judge 均分叫工具成功。当前 toolkit 新增任务，不能拼接原论文整体平均 |
| FDB-v3 | [2604.04847v1](https://arxiv.org/html/2604.04847v1) Table 2：Cascaded Whisper→gpt-4o→TTS，配 GPT-Realtime | Pass@1 **0.450 / 0.600**；tool F1、argument score、响应质量、turn-taking、latency 分开。真实人声 **100 examples**；mock tools；argument 和部分响应评分用 GPT-4o judge | L3 人声工具验证表。[v3 官方管线](https://github.com/DanielLin94144/Full-Duplex-Bench/tree/3e799c45a045256f47d5f1c9cda90157e2d2ec9e/v3) 可定位，LiveKit/provider/judge 尚未配置；固定录音不是动态人类用户 |
| EVA-Bench | [2605.13841v1](https://arxiv.org/html/2605.13841v1) Table 2/Figure 2，GPT-Realtime-1.5 与 Nova+GPT-5.4+Sonic cascade；Appendix B 系统设置 | EVA-A、EVA-X；pass@1、至少一次成功 pass@k、全部成功 pass^k 不互换。clean **213 scenarios / 3 domains / k=5**；扰动 **90 / k=3** 另列 | L3 后续；[官方框架](https://github.com/ServiceNow/eva/tree/8e101906119e1ddf3314d49913009f3a04327f0d)。本地 HF 仅 airline 入口，不能报三域 aggregate；provider 与 judge 依赖未配置 |

MASSIVE/ATIS/SNIPS 文本版、VoiceAgentBench、From Text to Voice、VGBench、Heard Enough to Act、MCR-BENCH 等继续留候选层，不增加本次资源下载或必跑矩阵。Qwen-Audio-Jev 未核验公开作者权重，不能用第三方同名 checkpoint 替代。

## 4. τ² voice 的版本选择

**锁定后续主轨为现有 commit `033bf082f48e4a3c6a1256208e93cf2ce9c3bb6f` 的 voice-v2**。原因：已有数据/环境桥以此版本固定，官方 [running-the-benchmark](https://github.com/sierra-research/tau2-bench/blob/033bf082f48e4a3c6a1256208e93cf2ce9c3bb6f/docs/running-the-benchmark.md) 明确给出同版 text-v1 与 voice-v2 的相同任务框架，可做真实语音—文本对照。

该 manifest 为六语言、三个域、每域 50 个固定任务，即 **18 个语言×域 cell、900 次任务运行/trial**；语言/persona binding revision 不同，必须记录。只先跑英语 retail 应称 voice-v2 cell pilot，不能称完成整个 benchmark。`--audio-native` 默认就是 voice-v2，执行时仍须显式指定 benchmark。`control`/`regular`、provider model ID、TTS voices、用户 simulator、judges 都纳入运行清单。发布方 persona voice ID 为内部账户资源，自建声音需标 override，不能承诺声音完全一致。

[原 τ-Voice 2603.13686v1](https://arxiv.org/html/2603.13686v1) Table 3/6 的英语 **278 tasks**、Clean/Realistic、GPT-4.1 用户 simulator 是历史文献栏。固定当前仓库也有 voice-v1 manifest，但“同名任务数”不足以证明重现论文旧运行栈。原论文与 voice-v2 不混排。

现有 [环境桥](../../experiments/tau2-environment-bridge.md) 只重放 TRAIN task 的参考动作、验证 DB 变化、重放一致性和反例；没有语音 provider、模型生成轨迹、完整 NL assertion/交互评分。它是启动门槛的一部分，不是 L3 学术基线结果。

## 5. 同骨干公平对照和资源归因

以下保留同骨干比较设计；当前执行以用户要求的音频端到端、冻结基座优先为准。权重训练暂停，串联/转写替换路线已退出执行范围。

- 历史 Qwen3-ASR→独立文本模型及人工 transcript 替换实验仅保留诊断证据，不再派发，不作为本项目端到端基线或教师提示优化收益。文献中此类方法仅列外部参考。
- 当前冻结对照使用本地 Qwen3-Omni-30B-A3B-Instruct：原生任务、派生 Jev 与离线教师优化分别报告；不新增 Qwen2.5-Omni-3B 下载。Qwen3-ASR 任务输出及标签读出保留历史诊断，模板能力边界见 [源码核查](../engineering/qwen3-asr-task-template-analysis-2026-11-10.md)。不同模型/接口分别报告，不将更换骨干算作提示优化收益。
- 冻结 Qwen3-ASR-0.6B 声学表征+任务头：原生闭集基线；要参加 L2 动态 schema 比较，必须显式增加读 schema 的条件路径。仅预测旧标签再映射不等于懂 rebinding。
- Qwen3-ASR-0.6B 生成式 SFT、类型化读出 SFT：同音频主干、同源训练数据和候选定义、同训练 seed 集；开放槽位共同使用同一槽值模块或明确拆分能力，不能让类型化方法只做选择而生成式方法承担所有抽取。
- 同预算 RL：相同初始化、有效 source groups、候选供给、奖励可见信息、训练/校准集；分别匹配样本预算、输出/计分 token、更新预算和设备时间，不能用一个“epoch 相同”替代算力公平。生成式与 typed 的可计费动作/采样单位差异必须报告。

本次复核 Qwen/Qwen3-ASR-0.6B-hf 的官方 revision **7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c**，文本 Qwen/Qwen3-0.6B 继承已验 **c1899de289a04d12100db370d81485cdf75e47ca**。不把 Omni 作为 ASR 版本替代，也不声称 0.6B 是遍历所有近期发布后的“最新唯一版本”。

大骨干、外部预训练/合成训练、API 系统设外部参考栏。SpeechBrain 的外部 ASR 预训练、AudioJev 的多任务数据、MAC-SLU SFT、SFC 内域 RL 都不能与本项目 real-only transfer 混成同预算榜。ASR 和文本模型的总参数、声学前处理、重复问题读出、生成成本都分账；只报小 head 参数不能代表整个系统大小。

## 6. 本次证据质量与未核验项

本次重新在线读取核心论文全文、官方 README/config/scorer/HF 元数据；PDF 用本地 `pdftotext -layout` 核表，HTML 表格逐行核查。GitHub API commit 查询返回限流后，以 `git ls-remote` 核验公开分支 SHA，未 clone 大仓库。初始 Python HTTPS 证书链失败、路径猜测的 404、API 限流等都保留，随后成功来源不掩盖失败。未认证任何论文作者的实测正确性。

已有 `.local/research/sources` 只作线索和对照，源登记明确 `prior_cache_original_retrieval`；缓存原检索时间不冒充此次访问。共享模型/语料的完成状态引用本地验收，本次没有重哈希大文件，没有启动 CPU/NPU/CUDA 模型。主要未核验项：

1. Audio2Tool 当前官方 scorer 与论文协议对齐、release loader、历史 prompt 与 checkpoint；SFC task weights、SFT exact config；SpeechBrain SLURP HuBERT 历史 commit 与配方不一致、FSC 实际训练覆盖项仍缺失。
2. 新增官方权重的本地加载兼容性、许可证随资产落盘和推理完整性；API 模型历史快照可用性。
3. SLUE gated 数据、Phase2 各任务专用 adapter；MMAU 新旧 release 对齐与 hidden scoring；MMSU/MINDS 作者子集 IDs。
4. CUDA 主机/环境尚未满足；NPU 四卡已完成基础算术与单卡音频 LoRA 更新验证；原生 verl 与 NPU vLLM 概率一致性尚未证明，见 [NPU 结果](../engineering/npu-audio-runtime-v1-result.json)。provider 预算、voices 和 judge 凭据仍待落实。
5. 全量公平训练预算和类型化开放槽值接口尚未冻结。文档已锁比较对象及禁止混排规则；这些执行配置保留 proposed，不覆盖旧实验。

需要主进程决策的是执行资源与预算、是否另立 SFC 内域重训轨道、Audio2Tool 论文协议与当前官方代码差异的适配范围、以及 L3 首个 voice-v2 cell/provider。无需这些决定即可审阅并采用当前学术锁表。

本次交付校验：58 条登记覆盖全部 16 项指定资源；JSON 可解析，稳定 ID 唯一，报告来源引用与结果定位字段齐全，公开 SHA 格式及此次成功响应的原文哈希通过检查。四个新增文件逐个检查 diff/空白错误；共享工作区其他变化由主进程独立产生，本次未修改。
