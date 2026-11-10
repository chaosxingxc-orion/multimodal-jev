# Qwen3-ASR 的模板与任务能力边界

结论：已检查的官方源码没有独立的SLU、Jev、问答或JSON任务模板。
Qwen3-ASR使用完整词表的自回归语言模型输出，并无禁止JSON字符的硬约束；
但公开调用协议和模型说明面向转写与语种识别，不能由架构推导出通用指令能力。
当前零训练任务基线应继续使用本地Qwen3-Omni。ASR作为未来训练底座仍可研究，
但需要实测或适配训练，不能把一次template修改当成已获得任务能力。

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
