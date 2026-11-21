# 实验记录

此目录随仓库的 `main` 分支同步。[实验台账](experiment-ledger.md)是研究实验和
运行版本的统一索引。一次实验按读取台账行、构建或复用数据、训练、评测、更新台账交付。
[五步规约](experiment-package.md) · [记录模板](TEMPLATE.md) · [完整报告](reports/README.md)

> 早期结论快照：2026-11-17 UTC。历史运行记录归档到 [`archive/`](archive/)，
> 原始文件在 `archive/raw/`；后续状态按证据时间记录在台账中。

实验领取、阶段通过、失败、停止和完成时更新台账。报告写明实际证据、成本与结论，修正按日期记录。实验假设与配置在实验目录维护。仓库首页仅在新增可核验的发布结果或复算方式发生变化时更新。[实验协议](../experiments/README.md)

## 有效结论

| 主题 | 结论 | 边界 |
| --- | --- | --- |
| SLURP SFT 格式学习 | 168/168 格式有效；非空目标正确 0/81，低于全 null 控制 87/168 | 只有格式学习，没有语义迁移 |
| vLLM 预处理对齐 | waveform 路径误差 0.16854；feature v4 单条 0.0000802、完整批次 0.0000566 | 固定录音/身份；工程一致性，不是训练收益 |
| FSC 数据人口 | 30,043 WAV；train/valid/test = 23,132/3,118/3,793 | 数据事实，不是模型成绩 |
| 早期原始音频策略 | FSC 186、SLURP 168 个开发决策全部格式无效 | 历史接口失败，已被概率头路线取代 |

## 归档

| 记录 | 总结 | 内容 |
| --- | --- | --- |
| 2026-11-17 早期实验结论 | [总结](archive/2026-11-17-experiment-results.md) | 零训练对照、早期 Stage1、GRPO、公开模型与效率 |
| 2026-11-10 开发集基线 | [总结](archive/2026-11-10-development-baselines-v1.md) | SFT、原始音频、vLLM 预处理、文本对照的当时状态 |

已归档的 FSC/SLURP 冻结 Omni 完整 TEST 记录属于 `experiments` 冻结基线家族，见
[冻结基线归档总结](../experiments/archive/frozen-baselines-v1.md)。
