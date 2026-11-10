# 本地 Qwen3-Omni 的 MMAR 原生基线准备

MMAR 是外部测试集，全部 1000 个问题用于独立报告，不用于教师反馈、提示选择、
奖励设计或训练。本项使用现有 Qwen3-Omni-30B-A3B-Instruct；不是复现 AudioJev
任务权重，也不替换为 Qwen3-Omni-Thinking。现有 AudioJev CUDA 入口保持独立。
当前只准备输入与评分、执行无权重 CPU 处理器预检查；没有新增 NPU 推理队列。

输入采用既有完整发布包
`041a543cc12bbf7b5bfa9e20e7c4a16ec3ca5b7efa5b2551957a3a02b93124e5`，
inputs SHA `d4ef0e308b5a27a163b83c06a9a0c89a6b2a64d8c4f7815dbee6d08a3187ad9c`。
只读取 question、choices、audio 及音频身份，不读取 references。原始候选的顺序、
重复值、空字符串和空白均保留。既有发布审计记录的四条空候选和六条非单候选答案
不按 gold 修复或排除，也不把所有答案都假定成单索引。

检查官方代码树 `3bce090a967db576c8ad433b290a8da582d6d2a0`：code 下只有
`evaluation.py` 和 `evaluation_rubrics.py`，没有可直接复用的推理模板。
本项目预先固定一般任务指令，而非声称重现论文未公布的 prompt：

```text
Listen to the audio and answer the question using the provided choices. Give only the answer text, without an option label or explanation.
Question: <original question>
Choices: <JSON array preserving original candidate strings and order>
```

system 使用 `You are a helpful assistant.`，user 中顺序为真实音频、上述任务文字。
答案输出为原生文本描述；不强制 JSON、不转写后调用独立文本模型。
后续生成固定 greedy、seed 17、最多 256 输出 tokens、上下文 4096，须先通过
全部输入预算检查和实际 NPU 容量验证再派发。若预算不成立，先保留失败并独立修订
协议；不依据测试答案/正确率修改提示或截断问题。

音频头显示 3–56 秒，中位数 20 秒，17 条超过 30 秒。`mmar_omni.prepare_request`
对同一字节缓冲核验 SHA、帧数、采样率和声道，显式平均声道、polyphase 转为
16 kHz，并按既有 vLLM 边界协议补不足一个 hop 的尾部零。记录转换和补零量，
不裁剪音频。`preflight_mmar_omni.py` 检查全部原始输入的真实 HF/vLLM 展开 ID、
完整音频帧、BF16 特征逐元素一致性以及 256 tokens 输出预留；不是模型前向。

原生生成评分复用发布者未改动的 `string_match`，传入原始输出，完整分母 1000。
不剥离围栏、修复 JSON、把标签映射成答案或删除截断输出。另报描述精确匹配、缺失、
截断及模态/类别分组。发布者 word-set matcher 本身允许某些额外文本，不把它误称为
严格字符串匹配。`evaluate_mmar.py --prediction-format raw-text` 使用新原生路径；
原概率读出评分仍为默认，二者分表。部分输出只能得到不完整诊断，不能作为正式成绩。

本地 6 项测试通过，覆盖原候选保留、标签字段拒绝、31 秒之后的真实波形保留、
音频篡改拒绝、原始输出评分与完整分母。它们不替代实际处理器预检查或 NPU 推理。
