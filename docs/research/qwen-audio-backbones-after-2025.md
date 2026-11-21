# Jev 下一代语音基座：2025 年以来 Qwen 公开权重筛选

核验日期 2026-11-21。范围为 Qwen 官方账号、能直接输入原始音频、公开可加载权重。阿里云可调用的 Omni API 与本地可训练的 checkpoint 分开。这里只核验架构与权重元数据，没有 Omni Jev 的 NPU 准入或效果结论。

| 权重及固定 revision | 原生音频判决价值 | 在本项目的用途 |
|---|---|---|
| [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B/tree/ae9e1690543ffd5c0221dc27f79834d0294cba00)，2025-03，Apache 2.0 | Dense Thinker，28 层、宽 3584，原生音频输入与指令理解。[官方记录](https://github.com/QwenLM/Qwen2.5-Omni)列出 2025-06 的 MMSU 表现；这个历史榜单让它值得进入语义任务候选，不代替本地结果。 | 首选训练扩容。全包张量约 20.83 GiB，音频判决只需 Thinker 的相关路径。单卡 BF16 推理实例有比 30B 更多的内存余量；在 181 未实测峰值或吞吐。 |
| [Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/tree/26291f793822fb6be9555850f06dfe95f2d7e695)，2025-09，模型卡标 Apache 2.0 | Thinker 的语言侧为 48 层、宽 2048、128 个专家、每 token 路由 8 个。音频编码器为 0.648B 参数。Instruct 与现有任务规约匹配。 | 本地已有；先作音频语义上限参照。按已存在的 15 个 safetensors 分片头相加，thinker.model 56.291 GiB，thinker.audio_tower 1.207 GiB，合计 57.498 GiB，不含视觉、语言输出投影、Jev 头、激活和运行时。不能由 A3B 推导单卡可承载双实例方案，也不能用磁盘大小声称 NPU 的实际峰值。 |
| [Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/tree/f75b40e3da2003cdd6e1829b1f420ca70797c34e)，2025-04，Qwen Research License | 原生音频，36 层、宽 2048。 | 需要更小的 Omni 或受限内存时备选。共享或商用边界应查本型号 LICENSE，不套用 7B 的许可证。 |
| [Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf/tree/7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c)、[1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf/tree/bcd2b5b7f32b480ab5790554cfa8347f246a14f3)，2026 | ASR 专项预训练。官方账号中找到这两种规模，原版与 Transformers 转换版不重复计数。 | 0.6B 是工程与成本基线，1.7B 是区分同系列容量效应的参照。没有证据证明 1.7B 的 Jev 决策能力已经到顶。 |

若“25 年后”严格指 2026 年起：在本次官方权重检索中，未核实到当年新发布、比 1.7B 更大的原生语音 Qwen checkpoint。[云端 Omni 文档](https://help.aliyun.com/zh/model-studio/qwen-omni)出现 3.5/3.8 Omni 服务；不能用服务名推定权重已开放。[Qwen3.8-27B 的公开配置](https://huggingface.co/Qwen/Qwen3.8-27B/blob/37561816e66b5b9153b1b64c10968c18c1e6f4a0/config.json)列出文本和视觉结构，没有音频塔；核对的 3.5-35B-A3B、3.6-27B 和 3.8-Flash-Next 也不能直接消费音频。自行训练音频桥接层会改变本次选型问题。Qwen2-Audio-7B 是 2024 年产品；TTS、ForcedAligner 和 Captioner 的预训练任务也不等同于 Jev 指令判决。

迁移时保留音频+任务定义 → 决策 marker 上下文表示 → 单层 Transformer → 分数的结构，适配 Omni 的音频 processor、token、hidden size、q/v LoRA 模块映射与服务端加载。Thinker 直接 forward；不生成转写、推理链或执行参数。各模型单独构造输入和 reference 缓存；共享原始录音、标签、训练列表、分割和决策协议。在 181 做同模型同权重的 BF16 训练/服务 logits 和一次更新后的加载校验。7B 首先尝试现有 6+1+1 拓扑，以实测容量决定 micro 和服务 batch，不假定现网参数可继承。

下一轮优先 7B 一个训练候选。沿用全量 clean TRAIN 和同一终点预算、Noul/Choice 分开报告 SLURP、MINDS、SIB-FLEURS 的最终 Top-1 macro-F1，附成对差、NONE 误拒/误接和每千录音的 NPU 卡分钟。30B 原生音频指令基线如果用选项打分，标明读出协议不同；只有装上同构、受训的 Jev head 后才作同协议模型比较。ASR 和 Omni 的差异同时包含预训练目标、模型架构和规模，不能当作 scaling law。若所要求的结论是纯容量效应，先用同一协议完成 0.6B/1.7B 对照。
