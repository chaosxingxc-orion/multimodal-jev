# MINDS-14无标签音频输入资产

2026-11-15已完成全部8168条原始音频物化，并由独立工具重新读取全部文件核验。
这是迁移输入准备，不是任务定义准入或模型评测。

本地资产目录：
`/home/xingchao/workspace/multimodal-jev/.local/decision-objective-research/minds14-audio-v1`。
本机没有/data；输出经挂载及容量检查确认为/home本地ext4，未写Ceph。
每个音频文件保留原始WAV字节、8kHz单声道；没有重采样、裁切或重新编码。
音频名为既有审计记录的不透明id，不含原路径中的意图名称。

`inputs.jsonl`仅含id、相对audio路径、编码文件hash、采样率、通道数、样本数和解码波形hash。
不含intent、language、参考转写、原始path或原始audio.path字段；后续actor仅消费音频与
统一候选规约，不能把评分侧审计记录合并进提示。该清单尚不含任务模板，不是正式推理输入包。
标签和语言分层信息仍保留在单独绑定的原始审计中，用于未来预登记评分。

生成器只读取parquet的path/audio列；核对全部14个源分片hash、8168条输入身份和原始
编码字节/解码波形后发布MANIFEST，并复查源分片未变化。失败时不发布成功manifest。
文件和最终目录提交保留本地fsync，不修改训练存储规约。

## 实际验证

- `tools/reports/materialize_minds_audio.py`完成14分片、8168条、24.597287小时物化。
- `tools/reports/verify_minds_audio_assets.py`独立重新读取全部文件，逐条复算编码hash与
  解码波形hash，核对完整人口、严格字段集合及无额外音频文件。
- 9项针对标签/转写泄漏及审计身份篡改的测试通过，耗时0.26秒。
- 全量验证见[minds-audio-materialization-independent-v1.json](minds-audio-materialization-independent-v1.json)。

manifest SHA256：`0ed2d25fb3f669b9040071da7d939909e6a9d40f85023b4de57654df4abf4152`。
输入清单SHA256：`f1a9cbd6b2b4c7256b2f6104ee33bd288a54bc587de95980fc23f19277025b38`。

## 剩余工作及研究边界

仍需冻结原生标签描述或语义定义协议、完整目标人口和评分，再执行共享16k前处理、
全量processor输入长度检查、14候选HF/vLLM一致性及实际冻结迁移推理。
不能把本次8k原始字节物化称为已完成16k模型前处理或模型接入。
类别边界证据仍以[定义来源记录](65-minds-native-definition-evidence.md)为准，
不因音频资产准备完成而把项目临时定义改称原作者规约。

目标仍为上游TRAIN上的项目冻结迁移评测，不是官方TEST或论文随机划分复现。
没有目标训练/调参、没有读取目标转写、没有新模型下载、没有NPU派发。
MINDS工作不改变当前SLURP GRPO→RLOO→FSC→clean TEST的固定队列；
1078项训练依赖及1081项TEST依赖本次复核全部保持一致。
