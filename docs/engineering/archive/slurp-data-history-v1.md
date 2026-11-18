# SLURP 数据/输入历史归档（2026-11-12 至 2026-11-15）

本文件收编早期 SLURP-Jev 输入格式、Span 规则对照、吞吐分析和短音频修复。原始 md 已移入
`archive/raw/slurp-data-history/`；数据清单、处理器和评测 JSON 收据仍在原路径保留。

## 做了什么

- 早期 SLURP-Jev 完整消费包、原生意图输入、全本体检视与评分接口。
- Span 规则对照（单次 Span、均衡互补规则、verl/vLLM 链）。
- 各类吞吐分析（global128/micro32、变长分桶、热缓存）。
- 短音频卷积边界修复与意图全候选验收。
- 旧单选分类分支的全量选择集终态。

## 有效结论

| 主题 | 结果 | 当前状态 |
| --- | --- | --- |
| 短音频修复 | 修复短音频卷积边界，意图全候选验收可继续 | 修复保留；当前输入仍须绑定波形 hash |
| 早期 Jev 输入/本体 | 完成输入、本体与评分接口审计 | 已被 clean source bundles 与正式发布身份取代 |
| Span 规则对照 | 规则输入 HF/vLLM 对齐；用于早期小样本实验 | 规则信用未证明稳定收益，见 [Span 总结](span-engine-v1.md) |
| 吞吐分析 | global128/micro32、分桶、热缓存等单点分析 | 已被 2026-11-16/17 声学+KV16 完整 100 步对照取代 |
| 旧单选分类分支 | 全量选择集终态保留 | 不是当前独立 sigmoid 成员判决动作空间 |

**结论：数据接口、短音频修复和输入审计仍有价值；旧的输入版本、吞吐数字和单选分支不能直接
用于当前实验。**

## 失败/过期路径

- 把旧吞吐单点或分桶 +1.05% 当作当前训练加速：最新对照以声学+KV16 为准。
- 用旧 Jev 输入版本代替当前 clean source bundles。
- 把旧单选 softmax 成绩与当前独立 sigmoid 集合指标互换。

## 保留证据

- `slurp-*-cpu-*.json`、`slurp-*-verified.json`、`slurp-two-stage-*.json` 等原收据。
- 当前活跃数据文档：`slurp-quality-hold-v1.md`、`slurp-clean-source-bundles-v1.md`、
  `slurp-selection-rule-controls-v1.md`、`slurp-portable-publication-v2.md`。
