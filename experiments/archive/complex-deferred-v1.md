# 复杂任务实验归档（SFC / τ² / SLURP 时间角色）

本文件收编本阶段明确 deferred 的复杂任务实验。原始 md 已移入
`archive/raw/complex-deferred/`；输入清单、评分契约和验收 JSON 收据仍在原路径保留。

## 范围

按 2026-11-17 协作指南，本阶段只做单音频、完整候选面板的简单任务。以下不进入训练、
评测或交付依赖：

- SFC：多轮音频、工具调用、类型化参数。
- τ² voice：交互状态、澄清、纠正与工具执行。
- SLURP 时间角色：时序状态先导与过程信用。

## 有效结论

| 任务 | 已确认 | 未确认/阻塞 |
| --- | --- | --- |
| SFC | 官方 split、5,534/2,590/771 turn 记录、输入/监督物理分离；8,895 请求输入及 TRAIN 标签绑定通过 | TEST 标签未打开；782 条双音频历史超出当前单音频 head 编译范围；未训练/未评测 |
| τ² voice | 环境桥接 fixture 与依赖解析完成；不修改共享 pinned 源码 | 无 LLM user、ASR、对话策略合规验证；不是学术基线复现 |
| SLURP 时间角色 | 三臂 SFT 诊断完成：base 0/142、SFT 80/142=56.34%、完整三类型 5/46 | 并行 Span 不是时序状态；不套 temporal GAE，不作选择集/TEST/学术基线 |

**结论：这些任务的接口、输入与评分边界可作为下一阶段起点，但本阶段不训练、不评测，
也不作为简单任务发布的依赖。**

## 失败/过期路径

- 把“输入已准备/配置可加载”当作任务能力或迁移收益。
- 把 SFC 未调用函数直接标为“不可用”：须先区分完整可适用集合与只记录一个被选择调用。
- 用单音频 head 强行处理双音频历史输入。
- 把并行 Span 当时间序列套 GAE 或过程奖励。

## 保留证据

- `sfc-*-verified.json`、`audio2tool-*-verified.json`、`mac-slu-*-prepared.json` 等原收据。
- 工程侧总结见 [复杂任务 deferred 归档](../../docs/engineering/archive/complex-tasks-deferred-v1.md)。
- 研究侧边界见 `docs/research/sfc-native-call-set-transfer-admission-v1.md`（若存在）。
