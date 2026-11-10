# Qwen3-ASR 的模板与任务能力边界

结论：已检查的官方源码没有独立的SLU、Jev、问答或JSON任务模板。
Qwen3-ASR使用完整词表的自回归语言模型输出，并无禁止JSON字符的硬约束；
但公开调用协议和模型说明面向转写与语种识别，不能由架构推导出通用指令能力。
当前零训练任务基线应继续使用本地Qwen3-Omni。ASR作为未来训练底座仍可研究，
但需要实测或适配训练，不能把一次template修改当成已获得任务能力。

## 官方训练目标提供了更明确的依据

[官方技术报告 v1，第 2.2 节](https://arxiv.org/html/2601.21337v1#S2.SS2)
在 ASR supervised finetuning (SFT) 段落明确写道：

> Specifically, we train the model to be an ASR-only model that does not follow
> natural-language instructions in the prompt, in order to mitigate instruction
> injection and instruction-following failures.

同段说明 system 中的 context 是转写背景知识。因而没有依据把它解释为
通用任务指令接口；冻结模型拒绝遵循“输出意图 JSON”符合其公开训练目标。
这比“尚未找到其他模板”更强，但仍不等于模型在任何输入下都不能生成 JSON 字符。
模型生成结构合法的 JSON、理解任务、预测正确字段是三个不同问题；
约束解码即使保证结构，也不能凭空保证后两者。

论文缓存 SHA 已与下载记录核对；固定 commit 的官方 prompt builder 已重新获取并核对
SHA。补充静态证据见 [audit v2](qwen3-asr-template-source-audit-v2.json)。

## 源码核查

- 本地`Qwen__Qwen3-ASR-0.6B-hf/chat_template.jinja`只收集system文字和音频。
  user中的文字不进入序列。静态哨兵渲染确认system保留、user文字丢弃。
- 官方`QwenLM/Qwen3-ASR`的`_build_messages`把context放system、音频放user；
  `_build_text_prompt`在指定语种时追加`language English<asr_text>`等assistant前缀。
- 镜像Transformers 5.14.1的`apply_transcription_request`把language直接放system；
  它没有显式读取prompt并构造上下文，不能假定与模型卡较新示例相同。
- 模型卡的`return_format="parsed"`是后处理生成Python字典，
  原始模型输出仍为`language English<asr_text>...`，不能算模型生成JSON。
- 模型实现`lm_head`投影到完整`vocab_size`并继承`GenerationMixin`；
  所见generation配置无JSON屏蔽。格式失败应理解为任务接口/行为未成立，
  而非证明模型从词表层面永远不能生成JSON。

源文件哈希、官方commit、实际运行版本、渲染结果见
`qwen3-asr-template-source-audit-v1.json`。没有加载新模型或新增推理实验。

## 非ASR任务只能使用实验性调用方式

官方 ASR 序列的语义是：

```text
<|im_start|>system
转写背景知识或专有名词（可为空）<|im_end|>
<|im_start|>user
<|audio_start|><|audio_pad|><|audio_end|><|im_end|>
<|im_start|>assistant
language English<asr_text>实际转写文本<|im_end|>
```

自动识别语种时，从 assistant 换行后开始生成；指定语种时，官方 wrapper 将
`language English<asr_text>`放入输入前缀。`<asr_text>`标识转写正文，
不存在源码已确认的 `<json>`、`<slu>` 或 `<jev>` 切换 token。
音频占位必须由 processor 和真实 waveform 配对处理，单独拼接字符串不会传入音频。

若继续诊断冻结ASR的任务能力，唯一已确认能传入完整任务文字的位置是system：

```python
messages = [
    {"role": "system", "content": task_instruction_and_dynamic_schema},
    {"role": "user", "content": [{"type": "audio", "audio": wave_16k}]},
]
rendered = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
)
inputs = processor(text=[rendered], audio=[wave_16k],
                   sampling_rate=16000, return_tensors="pt")
```

这是实验性任务输入布局，不是官方支持的非 ASR 模板。任务文字应包含完整字段、
取值集合和输出约束；不能只把它放到 user 的 text 项中。
本次同时静态渲染本地 1.7B 原始版和 HF 版，两者都丢弃 user 文字。
两者在此单轮输入下均渲染真实换行，输出序列一致。JSON 文件中的转义表现
不能直接当成最终 prompt；应先解析 JSON，再渲染 Jinja 检查实际结果。

实际序列为system任务文字、user音频占位、assistant生成起点。
不提供参考转写，不经过独立文本模型。自由任务生成不应把ASR语言头描述为
任务模式开关。`language English<asr_text>`只切到原生转写正文位置，
可作为预登记诊断条件；在它后面做有限标签logit选择属于自定义读出，
并不是官方SLU模板，也不等同自由生成成功。

## 对既有实验的影响

`infer_frozen_requests.py`已经把任务定义放在system（前置English）；
`direct_readout.audio_messages`同样把任务放system，并支持显式assistant前缀。
因此已有失败不能归因为任务文字放user而丢失。单独把English写入system
不是通用任务控制符，应与官方context/语言控制版本差异分开描述。

此前FSC零训练JSON结果为0合法，只能说明该模型、提示、解码和严格接口下
任务基线失败；不说明ASR识别率为0。有限标签读出也未建立可靠任务提升。
不在已看过的TEST上继续试模板或优化教师提示。若未来新增ASR适配实验，
在学习集预登记模板变体，并保持选择集/测试集隔离。
