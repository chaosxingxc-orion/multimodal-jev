# 实验协议与入口

> 状态快照：2026-11-17 UTC。本页是 `experiments/` 的阅读入口，只保留当前协议与有效结论。
> 历史实验已按家族收编到 [`archive/`](archive/README.md)，原始文件在
> `archive/raw/`；JSON 收据原地保留。归档协议不直接重新派发。

实验目录只保存协议、数据/模型清单和运行入口；结果摘要登记在 [wiki](../wiki/README.md)。
运行前固定数据划分、指标、阈值、随机种子、对照与失败判据；协议变更记录原因，不用结果
反向改写预注册条件。大文件、模型和完整输出保存在 Git 之外。

## 当前入口

| 文档 | 用途 | 边界 |
| --- | --- | --- |
| [学术横向比较冻结方案 v1](academic-baseline-comparison-v1.md) | 公开模型、原生任务、派生 Jev 与交互任务的分层横比协议 | 论文数字不等于本地复测；复杂任务表仅留作下一阶段资料 |
| [Qwen3-ASR 简单 Final60 零训练基座](slurp-simple-base-selection-v1.md) | 完整 9,226 条 clean SLURP selection 的零训练控制；含[机器可读结果](slurp-simple-base-selection-v1.json) | 不是官方 TEST，不是随机分类，不是训练收益 |
| [工程索引](../docs/engineering/README.md) | 当前主线、算法、效率与迁移证据的唯一工程入口 | 实验目录不复制实时状态 |

当前主线任务、数据与算法（全量 clean SLURP 成员判决、Stage1 + 双信用 GRPO、FSC/MINDS
迁移）以 [工程索引](../docs/engineering/README.md) 和
[协作指南](../docs/collaboration/simple-task-rl-engineering-guide-v1.md) 为准。

## 有效结论

| 主题 | 结论 | 证据边界 |
| --- | --- | --- |
| 零训练控制 | Qwen3-ASR 简单 Final60：154/9,226 = 1.6692%，56 个 LoRA B 矩阵为零，等价未训练基座 | 单类塌缩；不能据此认定没有声学信息 |
| Stage1 | 634 更新后 selection set-F1 73.8041%、top1 82.0724%、NLL 0.0187434286 | 不是 TEST、不是 RL 收益；详见工程索引 |
| 正式 GRPO | 容器正常退出；完整 11 点 selection 未收口 | 进程 exit0 不等于审核通过或任务收益 |
| 学术参考 | FSC SpeechBrain 3,774/3,793 = 99.4991%；SLURP Omni clean 原生意图 5,207/12,991 = 40.0816% | 前者是监督模型；后者是历史冻结基线；非同预算 |
| 训练效率 | 声学批处理 +8.4058%，KV16 再 +3.8489%，合计 12.5783%（100 步） | 单次顺序运行，未完成 fresh replay；不是任务收益 |
| 早期 Intent/Span/规则信用 | 工程链路可用，但没有持续分类收益，存在常假退化 | 见归档总结；不能把单步或末点当稳定收益 |

## 归档地图

| 家族 | 总结 | 内容 |
| --- | --- | --- |
| 冻结基线 | [frozen-baselines-v1](archive/frozen-baselines-v1.md) | Omni/Qwen3-ASR/FSC/SLURP 冻结推理、FSC p1 提示优化、早期迁移 |
| 早期 RL 工程 | [early-rl-engineering-v1](archive/early-rl-engineering-v1.md) | 首次 SFT/RL、算法预算、vLLM CPU、文本对照、NPU 诊断 |
| Span/BF16 | [span-rl-probes-v1](archive/span-rl-probes-v1.md) | typed Span、BF16 LoRA、单步 verl、pooling、规则信用 |
| 复杂任务 | [complex-deferred-v1](archive/complex-deferred-v1.md) | SFC、τ² voice、SLURP 时间角色；本阶段 deferred |

## 维护规则

1. 新实验先写协议与结论边界，再冻结配置；不把工程检查写成任务收益。
2. 协议被新决策取代时，在总结中记录有效结论与失败路径，原始 md 移入 `archive/raw/`。
3. JSON 收据是实验细节证据，原地保留；移动 md 后更新活跃文档链接。
4. 归档只移动文件，不删除；删除须另走保留清单与独立审核。
