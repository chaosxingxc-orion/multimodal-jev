# 研究计划

> 状态快照：2026-11-17 UTC。本目录保留课题范围与仍然有效的路线决策；历史协议已归档到
> [`archive/`](archive/README.md)。当前任务、算法、范围与发布边界以
> [工程索引](../docs/engineering/README.md)和
> [协作指南](../docs/collaboration/simple-task-rl-engineering-guide-v1.md)为准。

## 当前有效入口

- [课题范围与初步工作项](00-research-scope.md)：语音多模态 Jev 的研究目标、任务/数据方向与评估原则。
- [端到端语音任务路线修订](03-end-to-end-audio-task.md)：音频与任务定义直接进入同一语音多模态模型，
  禁止 ASR 转写后接独立文本模型。

## 当前范围摘要

- 只做单音频、完整候选面板的简单任务：SLURP 原生互斥意图成员判决；FSC/MINDS 用同一接口
  换候选定义做跨数据集验证。
- 训练主线为 Stage1 指令/候选绑定 + Stage2 双信用 GRPO；不再把 RL vs SFT 胜负作为发布条件。
- 复杂任务（SFC、Audio2Tool、多音频历史、多意图组合、多步判决）本阶段 deferred，
  不进入训练、评测或交付依赖。
- 具体数据人口、指标、算法与效率证据见 [工程索引](../docs/engineering/README.md)，本页不复制实时状态。

## 归档

| 总结 | 内容 |
| --- | --- |
| [历史研究协议](archive/historical-protocols-v1.md) | ASR→文本三阶段、冻结 Omni/TFRL、Omni SFC RL、数据 v2 设计 |

归档协议中的假设和数字只用于追溯；重新启用须另立预注册协议和独立配置。
