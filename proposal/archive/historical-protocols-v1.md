# 历史研究协议归档（2026-11-09 至 2026-11-12）

本文件收编已被当前执行路线取代的研究协议。原始 md 已移入 `archive/raw/`；这些文档
保留当时的假设、预算和设计，不作为当前派发指令。

## 内容与有效结论

| 文档 | 当时内容 | 有效结论 / 取代关系 |
| --- | --- | --- |
| `01-rl-research-protocol.md` | 三阶段 ASR→文本→RL、SFT vs GRPO/pair-GRPO、200 次更新、调度 RL | 端到端硬约束与当前简单任务指南已取代 ASR→文本路线、RL/SFT 胜负比较和旧预算；只保留预注册、来源隔离与区分性对照原则 |
| `02-frozen-baselines-tfrl-jev.md` | 冻结 Omni、远程教师 TFRL、xx-jev 三阶段，权重更新为零 | 已被 Qwen3-ASR 概率头训练主线取代；离线教师读取学习集真实反馈、不读选择集/测试的原则仍有效 |
| `03-omni-sfc-rl-design-v1.md` | Omni 端到端 SFC 多函数/多音频 RL 设计 | SFC/多音频/多步判决本阶段 deferred；同输入分组、完整分母、不按 gold 选题的原则保留 |
| `04-speech-jev-dataset-v2.md` | 语音 Jev 数据 v2 的 NGA 来源隔离与定义遵循设计 | v2 未生成；来源组隔离、请求/监督物理分离、缺标 mask 的原则已进入当前 clean SLURP 数据契约 |

## 失败 / 过期路径

- ASR 转写后再调用独立文本模型，替代在线音频 actor。
- 把 RL vs SFT 的胜负作为当前发布条件；当前只比较同一 RL 运行的起点与检查点。
- 把冻结 Omni 提示优化或 xx-jev 包当作当前权重训练收益。
- 把未生成的 v2 数据设计写成已发布数据。
- 把 SFC 多音频、多函数、多步交互纳入本阶段简单任务验收。

## 保留证据

- 当前硬约束见 [端到端语音任务路线修订](../03-end-to-end-audio-task.md)。
- 当前研究范围见 [课题范围与初步工作项](../00-research-scope.md)。
- 当前任务、算法与范围以 [工程索引](../../docs/engineering/README.md)和
  [协作指南](../../docs/collaboration/simple-task-rl-engineering-guide-v1.md) 为准。
