# 复杂任务 deferred 归档（SFC / Audio2Tool / MAC-SLU / 时间角色）

本文件收编本阶段明确 deferred 的复杂任务资料。原始 md 已移入
`archive/raw/complex-tasks-deferred/`；输入清单、评分契约和验收 JSON 收据仍在原路径保留。

## 范围

按 2026-11-17 协作指南，本阶段只做单音频、完整候选面板的简单任务。以下任务不进入训练、
评测或交付依赖：

- SFC：多轮音频、工具调用、类型化参数。
- Audio2Tool：真实人声工具调用、多步交互。
- MAC-SLU：跨任务扩展准备。
- SLURP 时间角色：时序状态先导。

## 有效结论

| 任务 | 已确认 | 未确认/阻塞 |
| --- | --- | --- |
| SFC | 8,895 请求输入及 TRAIN 标签绑定通过；评分接口与目标转换边界已定义 | TEST 标签未打开；782 条双音频历史输入超出当前单音频 head 编译范围，未派发 |
| Audio2Tool | 无答案输入索引、原生/Jev 输入、评分契约、容量预检查已准备 | 模型库卡片 16,843 queries / 36,421 audio files 与论文约 30,000 不一致；未训练/未评测 |
| MAC-SLU | 一次填充接口与训练数据/评分接口审计 | 仅准备，不派发 |
| 时间角色 | 先导协议与输入已定义 | 不进入当前简单任务验收 |

**结论：这些任务的接口、输入与评分边界可作为下一阶段起点，但本阶段不训练、不评测，
也不作为简单任务发布的依赖。** 详见研究侧
[`docs/research/sfc-native-call-set-transfer-admission-v1.md`](../../research/sfc-native-call-set-transfer-admission-v1.md)
（若存在）。

## 失败/过期路径

- 把“输入已准备/配置可加载”当作任务能力或迁移收益。
- 把 SFC 未调用函数直接标为“不可用”：须先区分完整可适用集合与只记录一个被选择调用。
- 用单音频 head 强行处理双音频历史输入。

## 保留证据

- `sfc-*-verified.json`、`audio2tool-*-verified.json`、`mac-slu-*-prepared.json` 等原收据。
- 关键原始文档：`sfc-jev-protocol-v1.md`、`audio2tool-jev-v1.md`、
  `sfc-native-full-test-v1.md`（均在 `archive/raw/complex-tasks-deferred/`）。
