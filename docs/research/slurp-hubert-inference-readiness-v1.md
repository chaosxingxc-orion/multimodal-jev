# SLURP 官方 HuBERT 推理接口与离线依赖

核查固定任务revision `6644fb49e209d2c4c3ea6bfd8accbe06b2df28c4` 的
`hyperparams.yaml`、`custom_interface.py` 和发布文件元数据；仅获取源码与
元数据，没有下载权重或执行模型。来源原文哈希见
[接口核查记录](slurp-hubert-interface-audit-v1.json)。

## 实际输入与输出

`CustomSLUDecoder.decode_batch(wavs, wav_lens)`直接接收音频，调用
`self.mods.hubert(wavs.detach())`，将编码结果与相对长度传给beam search，
最后用SentencePiece解码语义字符串。没有先生成转写或调用文本LLM，符合
项目端到端约束。输入为16kHz单声道；基础feature extractor启用波形归一化。
接口的encoder调用没有传入`wav_lens`，因此首次复测应使用batch1，避免增加
不同录音间padding所导致的行为变化；这不是已测出的batch数值一致性结论。

发布配置为768维编码器，128维token embedding，三层512维LSTM attention
decoder，58个输出token，beam80、temperature1.25、eos_threshold1.5，
BOS/EOS均为0，decode ratio上限10。输出是语义序列字符串，不能假定是
严格JSON。必须保留原文，并单独核对历史recipe如何将其映射成官方
scenario/action/entities；不能直接套用Omni严格JSON格式率进行方法排名。

## 不应误判为错误的两点

发布YAML的`hubert`对象实际引用旧路径
`speechbrain.lobes.models.huggingface_transformers.wav2vec2.Wav2Vec2`。
已固定的SpeechBrain `c0241278f93ef02cd5191f03caf4e5f8e1292bf6`存在旧命名空间
重定向；其wrapper通过AutoConfig/AutoModel按`model_type=hubert`选择模型。
因此旧路径或类名Wav2Vec2本身都不能证明加载错骨干。当前HuBERT包装类也
继承Wav2Vec2；是否与发布checkpoint完全兼容仍须真实严格加载验证。

## 默认离线加载缺口

任务仓库的核心文件共410,535,854字节：`hubert.ckpt` 377,575,130字节，
`model.ckpt` 32,722,503字节，tokenizer 238,221字节。发布metadata提供两份
checkpoint的LFS SHA256；这只是发布方身份，尚非本地文件校验。

YAML先构造`facebook/hubert-base-ls960`，之后Pretrainer覆盖微调encoder、
decoder及tokenizer。因此默认入口还需要基础HuBERT配置、feature extractor
和377,569,754字节的`pytorch_model.bin`，不能以“有任务checkpoint”声称已可
离线复测。本次固定基础仓库revision
`dba3bb02fda4248b6e082697eee756de8fe8aa8a`；配置明确12层、768维、12头，
前处理为16kHz/normalize=true。两组核心权重/tokenizer合计788,105,608字节，
不含小型配置、说明文件和缓存开销。未下载这些文件。

未来若用config-only构造后直接恢复任务encoder，以省去基础权重，属于另行
加载适配；需先证明每个参数都被原checkpoint覆盖且值一致，不能把缺失键
静默随机初始化，也不能直接称为未改官方入口。本轮未实现此适配。

## 复测还缺什么

1. 本地完整任务/基础资产及文件哈希，固定所有本地路径并禁止隐式联网下载。
2. 在隔离SpeechBrain镜像中验证旧路径解析、模型完整加载、checkpoint键和值，
   保留依赖版本。FSC在该镜像运行成功不能代替HuBERT加载验证。
3. 核对历史语义解析和官方scorer映射，在TRAIN上完成音频→原始语义→评分探针。
4. 冻结batch1/解码设置/资源时限后，执行完整13,078条real TEST，报告全部
   格式失败与官方SLU-F1，和作者75.10文献数字分列。

历史模型卡commit与HuBERT训练recipe的不一致仍保留，见
[历史源码追溯](speechbrain-history-v1.md)。即便后续权重复测成功，也不能
据此补写作者真实训练配置或声称已完成训练全过程复现。
