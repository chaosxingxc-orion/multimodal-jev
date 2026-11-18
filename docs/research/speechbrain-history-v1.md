# SpeechBrain 学术基线的历史源码追溯

2026-11-10 重新读取官方 GitHub commit API、完整递归 tree 和固定 SHA 原始文件。
9 次请求均成功，响应哈希、URL 和检索时间见
[来源记录](speechbrain-history-source-audit-v1.json)。仅下载元数据与源码，没有新增模型下载。

## FSC

模型卡短 commit `f1f421b3` 已解析为
`f1f421b3bb58dabc75d67c3fd5f6e3359943b927`（2021-06-02）。
该提交包含 `recipes/fluent-speech-commands/direct/hparams/train.yaml`、
`direct/train.py` 和 `prepare.py`，已读取三份原文。

配方使用 `speechbrain/asr-crdnn-rnnlm-librispeech` 的编码器，采样率 16000，
beam=80、temperature=1.25、eos_threshold=1.5，与本地权重复测登记的解码设置相符。
历史 YAML 默认 epochs=1、batch_size=16、lr=0.0003；这些是该提交的默认值，
没有训练命令覆盖项或作者日志，不能据此填成发布权重的实际训练设置。

历史 train.py 将解码语义与目标按空格拆分，累积 WER/CER；当前读取的代码没有
给出作者 99.60% accuracy 的独立计算过程。故本地 3774/3793（99.4991%）仍定义为
官方权重在固定当前推理环境的原生 joint intent 与序列精确匹配复测，不升级为
历史训练全过程复现，也不将 1-WER 当 joint intent accuracy。

## SLURP HuBERT

固定模型卡 revision `6644fb49e209d2c4c3ea6bfd8accbe06b2df28c4` 声明训练 commit
`aa018540`，并要求运行 `recipes/SLURP/direct/train_with_wav2vec2.py` 配合
`hparams/train_with_wav2vec2.yaml`。
短 SHA 已解析为 `aa0185408025e80f6c748d2c7af7fa96958c2231`（2021-02-17）。

GitHub 递归 tree 返回 truncated=false。该 tree 的 SLURP/direct 下只有
`train/train.py`、`train/hparams/train.yaml` 和准备文件，没有卡片引用的 wav2vec2
脚本/配方。实际历史 YAML 使用 LibriSpeech seq2seq ASR 编码器，非 HuBERT。
此前按当前目录结构读取历史文件的 404 不是 commit 不存在；现在明确了路径变化，
同时发现模型卡的 HuBERT 配方与所声明 commit 无法对应。

因此保留官方 HuBERT 权重复测目标与 75.10 SLU-F1 的作者报告；历史训练配方仍
待核实，不能用该 ASR 配方冒充 HuBERT 重训，也不能用当前 develop 配方补写历史。
本次没有断言模型卡为何产生不一致，没有验证发布权重的真实训练环境。

## SLURP历史tag补证（2026-11-13）

通过`git ls-remote`解析v0.5.11为
`d6bfe138a90dff3490f7196acbd9c62939289d46`、v0.5.12为
`fc15db47cebd191e4f1ba9bb9040f2ee469873e4`。两版均有HuBERT配方；v0.5.11
进一步固定SHA读取yaml、train.py、encoder wrapper及README，tag与SHA的yaml字节
相同。GitHub路径commit历史API返回403限流，未获得完整演化链，不宣称找到首次引入
该模型的提交。来源及校验见[补充审计](slurp-hubert-historical-recipe-audit-v2.json)。

固定README已经报告HuBERT intent87.54%、SLU-F1 75.10%，因此这一结果并非只在
当前develop文档存在；但不能据此证明HF发布权重由该tag训练。模型卡所声明aa018540
不含HuBERT配方的矛盾仍保留，历史训练命令与覆盖项尚未知。

配方默认real+synthetic、seed1986、35epochs、batch6、decoder LR3e-4、encoder
LR1e-5、卷积层冻结、encoder其余部分可训练、label smoothing0.1、速度95/100/105。
验证保存按语义序列WER选择，不等同最大intent accuracy或SLU-F1。这些是可读源码
默认值，不能填成发布checkpoint的实际训练设置。文件头写GRU，实际对象为三层512
LSTM attention decoder、128维embedding、58词表；结构判断以实例化字段为准。

历史encoder wrapper使用`F.layer_norm(wav,wav.shape)`和
`F.layer_norm(out,out.shape)`，均包含batch维。原始`extract_features`函数用numpy
layer_norm与代数编码器替身执行，同一首条波形在单条/两条batch下最大输出差2.75007，
说明该函数具有batch耦合。这不是HuBERT真实前向或精度测量，也不证明当前wrapper
与历史wrapper完全相同。首次官方权重复测继续batch1，同时核对实际加载wrapper和
归一化；禁止用改归一化或增大batch的结果冒称相同设置复现。

本轮只补源码证据，没有模型权重下载、模型推理或训练。788,105,608字节核心资产
未获本地加载证据，官方完整TEST复测仍未完成；不借FSC结果或作者分数填补本地结果。

## 配方引用论文与发布权重分开登记

读取配方引用的[arXiv:2111.02735v1](https://arxiv.org/pdf/2111.02735v1)全文，并对
第4页Table2渲染图逐项核对。文题为《A Fine-tuned Wav2vec 2.0/HuBERT Benchmark
for Speech Emotion Recognition, Speaker Verification and Spoken Language Understanding》。
PDF身份及9行结果见[论文审计](slurp-hubert-paper-source-audit-v1.json)。

| 来源/设置 | 意图准确率% | 槽相关指标% |
|---|---:|---:|
| 论文v1 PF-hbt-base：冻结CNN、微调Transformer | 87.51 | 75.32，文中标SF F1 |
| 论文v1 EF-hbt-base：整个encoder微调 | 87.44 | 75.06，文中标SF F1 |
| 论文v1 Frozen-hbt-base：冻结encoder，训练下游decoder | 68.74 | 57.06，文中标SF F1 |
| 已核查发布README/模型卡HuBERT | 87.54 | 75.10，发布方标SLU-F1 |

这些行不合并、不相互替换。特别是Frozen行仍有任务训练后的decoder，不能用于声称
“无训练基座”的表现。论文SLU为直接音频→语义序列，不是转写后交给独立语言模型，
符合端到端参考边界；但是序列生成槽内容与本项目S_present类型存在判决不是同一指标。

论文明确官方train/validation/test、HuBERT-base使用LibriSpeech960h预训练，SLU
encoder/decoder初始学习率1e-5/3e-4、Adam、beam80、无coverage penalty。正文没有
明确SLURP是否混synthetic，也没有给SLU种子数；SER/SV的五种子描述不得迁移至SLU。
论文称attentional GRU及字符级语义生成，而固定tag/发布配置实例化LSTM与58词表；
保留不一致，不能用后者补齐前者历史设置。论文未绑定精确scorer revision，SF F1的
具体实现仍不能仅凭数值改名，更不能当作槽位类型F1。

横向比较分为原始官方TEST的文献/发布权重定位、同一clean人口的匹配模型复测、
同一起点与固定人口的Stage1→RL曲线。clean筛除样本、内部TRAIN-selection和新增
语义标签均不继承论文分母，不能将跨协议差值称为RL增益。此次只归档论文，不派发模型。
