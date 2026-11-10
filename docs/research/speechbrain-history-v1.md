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
