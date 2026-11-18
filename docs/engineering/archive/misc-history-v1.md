# 其他历史记录归档

本文件收编不属于上述家族的零散历史审计。原始 md 已移入 `archive/raw/misc-history/`；
相关 JSON 收据仍在原路径保留。

## 内容与结论

| 文档 | 有效结论 | 边界 |
| --- | --- | --- |
| `actor-artifact-audit.md` | actor 产物保存/加载范围的审计 | 早期 actor 实现；当前以正式 runtime 与 LoRA 存储规约为准 |
| `end-to-end-readout-status-2026-11-10.md` | 冻结读出端到端状态快照 | 时间快照，不能当作当前能力 |
| `external-audio-overlap-v1.md` | 五个核心数据集音频文件哈希交叉核查 | 数据审计结论保留；不替代当前 release 身份 |
| `fsc-context-transfer-v1.md` | 当前概率头接 FSC 原生迁移输入的早期设计 | 已被当前 FSC 迁移矩阵与正式评分入口取代 |
| `fsc-jev-slurp-progress-20261110.md` | 2026-11-10 的 FSC-Jev / SLURP 进展 | 历史快照，不代表当前训练状态 |

**结论：这些记录提供数据/接口审计与历史状态，不作为当前发布或效果证据。**

## 保留证据

- `actor-artifact-audit-v1-result.json`、`fsc-history-reconciliation-v1.json` 等原收据。
- 当前 FSC 入口见 [工程索引](../README.md) 第 4 节。
