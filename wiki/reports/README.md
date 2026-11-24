# 实验报告

这里保存每次实验的完整报告及其修订。每份报告保留数据口径、复用关系、运行成本、评测分母、对照结果和独立的跨数据集结论。原始预测、模型和运行日志通过证据身份引用。[实验台账](../experiment-ledger.md) [报告模板](../TEMPLATE.md)

| 实验 | 报告 | 状态 |
| --- | --- | --- |
| 近期汇总 | [0.6B 训练、迁移与评测工程](recent-experiments-2026-11-22.md) | 含 204、205、206 已审核成绩及后续待办 |
| 方法复盘 | [知识卡、概率基线与强化学习](knowledge-cards-probability-and-rl.md) | Stage1 和 Stage2 优化机制、已审证据与跨数据集校准方案 |
| JEV-194 | [关闭与启用投影的配对评测](JEV-194.md) | 全量留出评测已审 |
| JEV-R64-S2 | [双信用 GRPO 末点评测](JEV-R64-S2.md) | SLURP TEST 和 MINDS 已审 |
| 零训练对照 | [SLURP selection](slurp-simple-base-selection-v1.md) | 已复算 |
| JEV-198 | [固定与动态 reference 的配对评测](JEV-198.md) | SLURP 和 MINDS 完整面板已审，跨任务证据待补 |
| JEV-201 | [新版联合 TPO](JEV-201.md) | SLURP TEST、MINDS、SIB-FLEURS 与 MMSU 末点评测已审 |
| JEV-204 | [全量单阶段联合 TPO](JEV-204.md) | 正式重启 212/212 步及三点原生评测已审；补充诊断待完成 |
| JEV-205 | [固定目标排序修正](JEV-205.md) | 34/34 步与五组评测已审；源域 selection 未达标 |
| JEV-206 | [逐类概率与双向对比奖励](JEV-206.md) | 34/34 步、源 TRAIN 校准及五组评测已审 |
| JEV-207 | [新构造的音频对与 actor 排序](JEV-207.md) | 训练与五组评测已审，源域未晋级 |
| JEV-208 | [Stage1 错例覆盖与当前概率间隔](JEV-208.md) | 102 步与五面板已审，未晋级；[错例分析](JEV-208-errors.md)、[三点验证](JEV-208-curve.md) |
| JEV-209 | [0.6B r128 知识卡与融合 TPO](JEV-209.md) | 混合版全 TRAIN 打分已审，第二轮正式第 144–146 步已审核；[真人版报告](../archive/JEV-209-real-only/report.md) |
| JEV-210 | [0.6B 多负卡 Stage1](JEV-210.md) | K=16，直接对照 209 混合版；CPU 配置与封包通过，待 NPU 验收 |
