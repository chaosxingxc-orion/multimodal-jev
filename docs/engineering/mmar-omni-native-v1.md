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

全量 CPU 预检查已从源码 `52756e8` 派发，容器
`34cf1ae6aa2605358358669e7d08d02255c5dcd436a3a739de47fd69ef9f89c4`，
固定原镜像、4 CPU/20 GiB、1800 秒加 30 秒，无网络和设备。仅复制已核验的七个
processor/tokenizer 配置文件，未挂载模型权重或参考标签。协调者实际 inspect
核对命令、环境、资源、挂载及 118 份源码与 Git，确认容器运行中；见
[启动核验](mmar-omni-native-preflight-v1-launched.json)。
运行结果及全部 1000 条的独立覆盖核验仍待完成，不将派发成功视为处理器一致性通过。

## 全量处理器预检查完成

上述精确容器已退出 0，1000/1000 原始 ID 完整且唯一；17 段超过 30 秒的音频
均按原长度处理，全部 HF/vLLM 展开 token 与 BF16 特征在运行时逐元素一致。
输入长度 104–815 tokens，最大输入加 256 输出预留为 1071，均在 4096 内。
实际处理耗时 289.91 秒，未加载权重、读取标签或调用推理引擎生成答案。

协调者重新 inspect 并核对实际命令/挂载/环境/资源、118 份源码与 Git、七个
processor 资产、全部逐条 ID、采样长度、补零量、音频帧和 token 预算。
见[独立完成核验](mmar-omni-native-preflight-v1-verified.json)。
特征张量没有保存，独立核验可复算覆盖和帧数，不能从记录重新复算特征数值；
特征相等结论来自已核验源码中的运行时比较。NPU 完整生成、任务评分及与 Jev
格式的对应比较仍未执行；这次成功不能代替模型能力验证。

## 原生推理入口与容量边界

`tools/jev/infer_mmar_omni.py` 提供独立 `--capacity-only` 模式与完整模式。
容量模式从已接受的 1000 条处理器记录选最大展开 token 数和最长音频，按原顺序
解决并列、去重，不读取标签。当前两项最大值均落在
`SQadcm_dwEM_00-01-05_00-02-01`：815 tokens、896000 个 16 kHz samples
（56 秒）。容量模式仅生成一个 token，不算任务答题或完整解码性能验证。

完整模式保持原始 1000 条顺序，greedy/seed17/最多256输出tokens。两种模式均固定
TP2、BF16、4096上下文、单音频、单并发、eager、显存比例0.85，关闭prefix cache；
没有声称关闭所有多模态缓存。原本地模型资产在运行前后完整校验。
每条重建真实音频和 prompt，与已接受处理器记录核对音频收据、帧数、展开ID哈希，
随后核对 vLLM 返回的实际 prompt IDs。先写入调用记录，再生成；完整保存原始响应，
校验失败也保留输出和异常，不继续运行剩余问题或自动重试。截断输出保持原文。

完整模式必须提供已完成的容量容器实际 inspect 及固定哈希，核对 NPU6/7、镜像、
命令、资源、挂载、源码、模型、采样预算及边界记录，并把预测摘要与保存的原始
响应和调用记录逐项比较。inspect JSON 本身不是外部来源证明，协调者必须实际
查询对应容器。当前只实现入口及校验，没有创建容量容器或新增NPU队列，也没有
MMAR生成结果。后续容量运行按16CPU/128GiB/8GiB共享内存准备，并接到既有队列之后，
不能绕过当前任务的设备占用或把成功创建容器算作验收通过。

33项相关本地测试通过；容量容器、结果和异常分支使用合成fixture，未冒充实际NPU
验证。另以真实完整输入及处理器文件执行了人口覆盖/哈希准入，确认1000条与上述
唯一边界。下一步是固定镜像运行入口预检查、安排容量探针及完整推理，再独立验收评分。
