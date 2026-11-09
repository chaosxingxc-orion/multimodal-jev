# 模型与可改造接口

核验范围、访问时间及内容哈希见 [来源说明](05-sources.md)。下文“已核验”指官方文档、模型配置或源码层面的核验，不代表已运行。

## 1. Qwen 系列应该怎样区分

| 系列 | 本次核验状态 | 对本课题的定位 |
| --- | --- | --- |
| Qwen3-ASR-0.6B / 1.7B | 官方仓库、模型卡、配置、权重 revision、训练脚本均可访问；Apache-2.0 | 首选可训练 ASR 参考骨干 |
| Qwen3-ASR-0.6B-hf / 1.7B-hf | 官方原生 Transformers 格式；不是新一代架构 | 便于标准接口开发的候选运行路径 |
| Qwen3-ForcedAligner-0.6B | 官方非自回归语音—文本对齐模型，11 种语言 | 时间定位和数据处理辅助；不负责语义决策 |
| Qwen2-Audio / Qwen2.5-Omni | 开放音频指令理解系列；AudioJev 基于 2.5-Omni-3B | 直接音频生成式理解和已有决策模型对照 |
| Qwen3-Omni-30B-A3B | 官方 Instruct、Thinking、Captioner 权重可见；Omni 已有工程经验 | 较大语音理解参考；不能只按 3B 激活量估算加载内存 |
| Qwen-Audio-3.1-ASR-Flash-Next | Qwen-Audio-Jev 论文描述了其结构；本次未找到官方开放权重 | 跟踪文献，不作为可立即复现的基线 |
| Qwen-Audio-Jev | 已读论文 v2；未核验到作者指向的开放权重和完整训练代码 | 直接相关研究，不能与社区 `audio-jev-qwen` 混称 |
| Fun-ASR-Nano / MLT-Nano | QwenAudio / FunAudioLLM 另一公开系列；Nano 与 MLT 能力不同 | 中文及多语言 ASR 横向对照 |

“最新”限定为本次查询到的官方开放权重。QwenLM、QwenAudio、FunAudioLLM 的发布线并不完全同名；论文、云服务、开放权重、格式转换版分别登记，不按模型名猜测可用性。[M1–M5、M10；编号见来源说明]

## 2. Qwen3-ASR 的具体版本

官方仓库记录 2026-01-29 发布，2026-06-26 增加原生 Transformers 支持。此次查询的模型 revision：

| 模型 ID | revision |
| --- | --- |
| `Qwen/Qwen3-ASR-0.6B` | `5eb144179a02acc5e5ba31e748d22b0cf3e303b0` |
| `Qwen/Qwen3-ASR-1.7B` | `7278e1e70fe206f11671096ffdd38061171dd6e5` |
| `Qwen/Qwen3-ASR-0.6B-hf` | `7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c` |
| `Qwen/Qwen3-ASR-1.7B-hf` | `bcd2b5b7f32b480ab5790554cfa8347f246a14f3` |
| `Qwen/Qwen3-ForcedAligner-0.6B` | `c7cbfc2048c462b0d63a45797104fc9db3ad62b7` |

参考代码 `QwenLM/Qwen3-ASR` HEAD：`7c6daf77a2421100f5fb066495372c00129d39ff`。这些是源码/模型仓库版本标识；已下载资源的字节校验状态见 [本地资源](06-local-assets.md)，不是经过兼容性测试的环境锁。

模型支持 30 种语言及 22 种中文方言，官方同时提供离线、流式识别和上下文 biasing。论文的吞吐、TTFT 均受硬件、并发、音频长度、vLLM/CUDA Graph 配置影响，不能换算成我们 NPU 或低并发任务的承诺。

## 3. 静态结构与潜在接入点

技术报告 §2.1 和官方 `config.json` 表明：128 维 Fbank → AuT 音频编码器及下采样 → 语言空间映射 → Qwen3 解码器 → ASR 词表输出。报告给出的编码帧率为 12.5 Hz；不应以这个平均值替代 processor 的实际长度计算。[M1–M3]

| 属性 | 0.6B 命名版本 | 1.7B 命名版本 |
| --- | --- | --- |
| 音频编码器层数 | 18 | 24 |
| 音频隐藏维度 | 896 | 1024 |
| 投入语言空间的维度 | 1024 | 2048 |
| 语言解码器层数 | 28 | 28 |
| 语言解码器类型 | Qwen3 attention | Qwen3 attention |
| 文本词表大小 | 151,936 | 151,936 |
| 配置最大位置数 | 65,536 | 65,536 |

名称中的 0.6B / 1.7B 不应直接当作包含所有音频组件的精确总参数量；报告另列音频编码器约 180M / 300M。实际训练与显存规划应按加载组件计数。65,536 是配置值，不是本课题已验证的长语音有效上下文。

已经检查的原始后端源码：

- `Qwen3ASRThinkerForConditionalGeneration.get_audio_features()`：根据 feature mask / 长度取得音频编码结果。
- `forward()`：把音频特征写入音频占位 token 的 embedding，处理位置与 attention mask，再调用语言模型及 `lm_head`。
- `Qwen3ASRThinkerTextModel`：可取得 `last_hidden_state`；在答案开始前的位置读出，是增加分类/候选头的候选位置。
- `processing_qwen3_asr.py`：特征长度和展开后音频 token 数必须一致；不能仅拼接文本 token 后忽略音频 mask。
- 官方 SFT 使用音频—转写 JSONL，屏蔽 prompt 的标签，只监督输出；这份训练脚本没有自动完成 Jev 的任务定义监督或槽位设计。

源码中的 `classify_num` 分支不等于一个已训练好的 Jev 分类头。独立头、共享 label-token logits、候选匹配头应作为不同方案比较。ASR 的 audio tower 冻结后可考虑缓存其输出；语言层 KV 是否可复用则取决于任务定义的位置与完整前缀内容，不能按音频相同直接复用。

**最重要的适配风险：**报告 §2.2 明确写道 ASR 后训练使模型不遵循一般自然语言指令。因而“有 Qwen3 语言解码器”不保证“能读运行时函数定义”。需要先做冻结表征探针，再比较读出头训练、投影/解码器适配及有限音频解冻。只改输出格式而没有定义遵循测试，无法证明实现了 Jev。[M1]

## 4. 两套环境不能混装成一个未验证组合

| 路径 | 当前来源要求 | 适用范围 |
| --- | --- | --- |
| 官方 `qwen-asr` 原始后端 | 源码 package 0.0.6；`transformers==4.57.6`、`accelerate==1.12.0`；可选 `vllm==0.14.0` | 复用原始源码、官方 SFT 和推理接口 |
| 原生 `-hf` 后端 | 模型卡声明 `transformers>=5.13.0`，采用标准 multimodal 接口 | 标准加载与框架集成 |

先单独验证两条路径的 processor、结构、mask 与结果，再选一条固定。当前项目 `.venv` 只是轻量开发环境，不表示已安装这些训练依赖。官方演示以 CUDA 为主；本次未验证 Ascend 算子、混精度、attention 实现或 vLLM-Ascend 适配。已有 Omni 能运行不构成 Qwen3-ASR 已适配的证据。

## 5. 值得保留的模型对照

| 模型/方案 | 可借鉴能力 | 主要限制/实验角色 |
| --- | --- | --- |
| Qwen2.5-Omni-3B / 7B | 原生音频条件指令理解；AudioJev 和 SFC 的基座 | 3B 衍生 AudioJev 使用 Qwen Research License；逐个模型核验许可，不能全系列统一推断 |
| Qwen3-Omni-30B-A3B | 较强多模态理解，workspace 有工程基础 | 成本较高；不同规模成绩不能归因为读出机制 |
| SenseVoiceSmall | ASR、语言、情感和声音事件的轻量多任务接口 | 固定任务标签不等于运行时自然语言定义 |
| Fun-ASR-Nano / MLT-Nano | 约 800M 系列，中文/方言或 31 语言覆盖 | Nano 与 MLT、HF 与 ModelScope 工件能力可能不同；官方特别提示时间戳权重差异 |
| Parakeet-TDT-0.6B-v3 | 高效 ASR、25 种欧洲语言，CC-BY-4.0 模型卡 | 非中文主基线；不提供通用 Jev 语义契约 |
| Canary-Qwen-2.5B | 语音编码器与 Qwen 组合的开放对照 | 当前卡片强调英语；不能据底层 Qwen 推断多语言 ASR |
| WavLM / HuBERT / wav2vec 2.0 | 冻结表征 + 轻量头，较低成本的必要基线 | 不天然读取动态候选定义；可用于区分声学表征收益与大语言模型收益 |
| SALMONN | 结合语音和通用音频编码器，连接语言模型 | 早期大音频模型路线，不是小型 ASR 决策模型 |
| Ultravox | 连续音频特征接入 LLM，音频输入到文本/工具 | 开放模型与托管实时服务功能分别核验 |
| Moshi / MiniCPM-o / NemotronLabs VoiceChat | 流式或全双工会话、实时交互；VoiceChat 有原生工具调用 | 更适合作为后续交互前端；不能用离线 ASR 直接替代其流式状态 |
| Apriel Voice 27B Preview | 最新语音推理与工具调用研究 | 本次读取论文摘要，未完成权重、许可证与服务复现核验；列作跟踪项 |

Whisper 可作为熟悉的 ASR/冻结编码器控制（Prosodia、SLAM-ASR 均使用其编码器），但本项目的主要模型选择仍以用户指定的 Qwen 开源 ASR 为中心。完整来源见 [全部来源索引](05-sources.md)。

## 6. 当前选择建议

先固定 0.6B，比较“冻结编码器轻量头”“已有 label-token 输出”“任务条件专用头”三种读出。1.7B 用于确认规模效应；AudioJev 作为可复现的直接先行模型；Qwen2.5/3-Omni 作为生成式音频理解参照。把模型大小、可训练参数、音频处理成本、prefill、决策头与槽位生成成本分别报告。
