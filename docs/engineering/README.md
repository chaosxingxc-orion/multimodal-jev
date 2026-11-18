# 工程索引

> 状态快照：2026-11-17 UTC。本页是 `docs/engineering/` 的阅读入口，只保留**当前有效的结论与决策**。
> 历史设计、失败探索和逐次运行记录收编在 [`archive/`](archive/README.md)，原始收据（JSON）原地保留。
> 运行事实以冻结包、配置、容器/进程回执和独立审核结果为准；本页数字是快照，不是实时状态。

## 1. 当前工程基座与实验设置

| 项目 | 当前值 | 权威入口 |
| --- | --- | --- |
| 任务 | 单音频、完整候选面板、SLURP 原生互斥意图成员判决；FSC/MINDS 用同一接口换候选定义 | [联合入口](context-prob-v1-baseline.md)、[协作指南](../collaboration/simple-task-rl-engineering-guide-v1.md) |
| 模型 | Qwen3-ASR-0.6B-hf；q/v LoRA rank4/alpha8；共享一层 Transformer 概率头（hidden1024、8 heads、FFN1024、residual0.1、dropout0） | [runtime](runtime.md) |
| 输出 | 每候选独立 sigmoid 概率；主决策为阈值 0.5 的候选集合，另报 top1；不生成文本 CoT、不做 ASR→文本模型级联 | [任务基线](context-prob-v1-baseline.md) |
| 训练 | Stage1 指令/候选绑定 634 步 → Stage2 双信用 GRPO 634 步；RLOO 不再作为运行分支；PPO-critic 仅设计预留 | [协作指南](../collaboration/simple-task-rl-engineering-guide-v1.md) |
| 并行/精度 | verl FSDP DP4、vLLM TP1；BF16 执行，FP32 master/Adam/loss | [网络与运行时](runtime.md) |
| 数据 | SLURP clean TRAIN 40,513、selection 9,226、TEST 12,991；FSC native31 3,793；MINDS native14 8,168 | [数据契约](data-contract.md)、[质量约束](slurp-quality-hold-v1.md) |
| 存储 | Ceph 模型/数据只读；热状态写本地盘并保留 fsync | [存储规约](runtime-storage-policy.md) |
| 监控 | 普通远程观察固定 900 秒；超时不等于训练失败 | [监控规约](monitoring-and-goal-continuation.md) |
| 本阶段范围 | 只做简单任务：效率、SLURP 源任务收益、FSC/MINDS 跨数据集迁移；复杂任务全部 deferred | [协作指南 §1](../collaboration/simple-task-rl-engineering-guide-v1.md) |

## 2. 当前有效结论

| 主题 | 有效结论 | 证据边界 |
| --- | --- | --- |
| Stage1 | 完成 634 次更新；完整 9,226 selection：set-F1 73.8041%、top1 82.0724%、macro-F1 77.4315%、NLL 0.0187434286、Brier 0.0046499592 | 不是 TEST、不是 RL 成绩；起点是新初始化概率头，不能称原生 ASR 零样本 |
| 正式 GRPO | 原 GRPO 容器 `e564df04` 正常退出；完整 11 点 selection 尚未收口 | 进程 exit0、TRAIN reward 上升均不等于任务收益；以收据为准 |
| 双信用 GRPO | 锁定算法 `dual_credit_grpo_floor_clipped_v1`，联合臂每分量 G=16；v3 整合包固定镜像 CPU 8 项检查通过 | CPU 导入/封包通过不等于 NPU 训练或效果；剩余放行条件是实际音频对齐、DP4 连续更新与恢复、全 TRAIN 数值覆盖 |
| 训练效率（最新） | 100 步对照：声学批处理 +8.4058%，KV16 再 +3.8489%，合计 **12.5783%**；三臂均 `finite_dp2_efficiency_audited` | 单次顺序运行，`fresh_process_replay_verified=false`；不含启动/收集/评测，不是任务或迁移收益 |
| 目标域一致性 | service32 的 FSC 有限面板 exact 通过；MINDS 259 条中 10 条差异（最大 prob 0.02438146、logit 0.21875，top1 翻转 0、阈值翻转 0） | 按失败停止，未执行 SLURP TEST；不能靠放宽 exact 条件宣布准入 |
| 学术参考 | FSC SpeechBrain 官方权重 3774/3793 = 99.4991%；SLURP Omni clean 原生意图 5207/12991 = 40.0816% | SpeechBrain 是 FSC 监督模型，不是零样本；Omni 是历史冻结基线；非同预算比较 |
| 跨数据集迁移 | FSC 3,793 目标输入与 MINDS 8,168 输入/processor 缓存已就绪；FSC 同人口参考比较入口已验收 | 尚无 Jev 实际迁移成绩；MINDS 是上游 TRAIN 外域评测，不称官方 TEST |
| 早期 Intent/Span 探索 | 多步 RL/规则信用在有限面板上工程通过，但未形成持续分类收益；存在常假退化 | 详细曲线和失败分解见归档总结；不能把单步或末点当作稳定收益 |
| Omni 基线 | 历史冻结基线与工程诊断保留；不作为当前主线或发布依赖 | 归档原因见 [Omni 总结](archive/omni-baseline-v1.md) |
| 复杂任务 | SFC、Audio2Tool、多音频历史、多意图组合、多步判决全部 deferred | 不代表已否定其价值，只是不占本阶段工作；资料归档见 [复杂任务总结](archive/complex-tasks-deferred-v1.md) |

## 3. 决策消解（冲突时以最新为准）

| 主题 | 最新决策 | 被取代/降级的旧说法 | 消解方式 |
| --- | --- | --- | --- |
| 算法主线 | 2026-11-17 协作指南锁定 `dual_credit_grpo_floor_clipped_v1`；RLOO 不作为当前运行分支 | [任务基线](context-prob-v1-baseline.md) 仍描述 GRPO + RLOO-clipped 同预算对照 | 保留旧基线作为发布身份与历史证据；新实验、新阅读以协作指南为准 |
| 阶段范围 | 只做简单任务；复杂任务 deferred | 早期文档中的 SFC/Audio2Tool/复杂 Q 推进计划 | 复杂任务原文归档，不再作为交付依赖 |
| 效率 runtime | 声学+KV16 是当前最快候选，但因 MINDS 概率一致性失败**未准入** | 早期“分桶 +1.05%”“声学 8.41%”阶段性数字 | 以最新 12.5783% 与失败证据为准；正式训练仍待准入 |
| 入口权威 | `AGENTS.md` + [任务基线](context-prob-v1-baseline.md) 管长期规约与发布身份；协作指南管跨团队当前范围 | 旧入口中过期的“运行中/排队中”状态 | 状态一律现场核验；本页不复制实时进程状态 |
| 存储/监控 | Ceph 只读、本地 fsync；900 秒观察 | 无冲突 | 维持不变 |

## 4. 活跃文档地图

**入口与规约**

- [任务与算法基线](context-prob-v1-baseline.md)
- [跨团队简单任务协作指南](../collaboration/simple-task-rl-engineering-guide-v1.md)
- [runtime](runtime.md)、[存储规约](runtime-storage-policy.md)、[数据契约](data-contract.md)
- [本地宏](local-config-macros-v1.md)、[日常操作归并](routine-operations-v1.md)、[监控与续跑](monitoring-and-goal-continuation.md)
- [SLURP 质量约束](slurp-quality-hold-v1.md)、[LoRA checkpoint 存储](lora-checkpoint-storage-policy.md)

**当前主线（context / 双信用 / 效率 / 迁移）**

- [声学批处理与 KV16 效率](context-acoustic-bucket-efficiency-v1.md)
- [DP2 整合](context-macro-dp2-integration-v1.md)、[正式候选](context-macro-formal-candidate-v1.md)、[正式启动](context-macro-formal-launch-v1.md)
- [源评测计划](context-macro-source-evaluation-plan-v1.md)、[评测准入边界](context-macro-evaluation-readiness-v1.md)
- [迁移候选宽度一致性](context-transfer-macro-parity-v1.md)、[全量迁移输入](context-transfer-portable-full-inputs-v1.md)
- [全量迁移部署](context-full-transfer-deployment-v1.md)、[全量评分](context-full-transfer-scoring-v1.md)
- [RL selection 入口](context-rl-selection-runbook-v1.md)、[selection 曲线](context-selection-curve-v1.md)
- [reference 复用](context-reference-reuse-plan-v1.md)、[连续窗口效率](context-reference-epoch-windows-v1.md)
- [双信号整合包 v3](context-dual-integrated-package-v3.md)、[双信号运行](context-dual-training-operations-v1.md)、[独立审核](context-dual-training-audit-v1.md)
- [数值协议修复](context-ranking-numerics-v2.md)、[损失分项验收](context-loss-metrics-native-validation-v2.md)
- [批量 rollout 候选](context-batch-probe-prepared-v1.md)、[整批 IPC 候选](context-bulk-ipc-candidate-v1.md)、[推理实例 overhead](context-replica-overhead-analysis-v1.md)

**FSC / 数据 / 运行时**

- [FSC 外部参考比较](fsc-external-reference-comparison-v1.md)、[FSC 官方评分](fsc-transfer-official-scoring-v1.md)、[FSC 迁移矩阵](fsc-dqp-transfer-matrix-controller-v1.md)
- [SLURP clean 来源包](slurp-clean-source-bundles-v1.md)、[selection 规则对照](slurp-selection-rule-controls-v1.md)、[评分路由](slurp-score-route-custody-v1.md)
- [BF16 Span runtime](span-bf16-runtime-profile-v1.md)、[verl/vLLM/LoRA 对齐](verl-vllm-lora-span-alignment-v1.md)、[NPU logprob 路径](verl-npu-logprob-source-v1.md)
- [单次 Span 契约](single-pass-spans-v1.md)、[四卡 FSDP 语音](span-fsdp-audio-npu4-v1.md)、[恢复接口](span-resume-v1.md)

**操作与保留**

- [保留库存](retention-inventory-v1.md)、[保留编排](retention-pipeline-v1.md)、[产物保留](artifact-retention-v1.md)、[归档编排](archive-retention-pipeline-v1.md)
- [收集编排](collection-trajectory-pipeline-v1.md)、[配置化收集](configured-training-collection-v1.md)、[数据采样](dataset-sampling-v1.md)
- [可移植配置校验](portable-config-verifier-v1.md)、[RL 配置](rl-experiment-portable-configs-v1.md)、[迁移配置](transfer-portable-configs-v1.md)
- [主机 Ceph 崩溃记录](host-ceph-crash-20261113.md)、[磁盘清理](host-disk-cleanup-20261116.md)、[NPU175 分配](npu175-allocation.md)

## 5. 归档与失败探索

归档入口见 [`archive/README.md`](archive/README.md)。当前按主题收编：

| 归档 | 内容 | 保留的有效结论 |
| --- | --- | --- |
| [Omni 基线](archive/omni-baseline-v1.md) | Omni/冻结基线的加载、K/V、采样、概率诊断 | Omni 是历史冻结基线；数值探针结论不再进入当前发布 |
| [Span 引擎](archive/span-engine-v1.md) | Span/verl/vLLM/TP4/FSDP 早期工程与规则信用 | 工程探针通过；规则信用未证明稳定决策收益 |
| [Intent RL](archive/intent-rl-v1.md) | 意图有限支持 D/Q/P 多步探索 | 工程链路可用；无持续分类收益，存在常假退化 |
| [Context RL 历史](archive/context-rl-history-v1.md) | 早期 context 候选、actor、缓存、吞吐、诊断 | 为当前双信用/效率线提供前置证据；过期候选不再执行 |
| [SLURP 数据历史](archive/slurp-data-history-v1.md) | 早期 Jev 输入、Span 规则、吞吐、短音频修复 | 数据接口与吞吐经验保留；旧输入版本不再使用 |
| [复杂任务 deferred](archive/complex-tasks-deferred-v1.md) | SFC、Audio2Tool、MAC-SLU、时间角色 | 范围与接口结论保留；本阶段不训练/不评测 |
| [其他历史](archive/misc-history-v1.md) | actor 审计、端到端读出、跨数据集音频重叠 | 审计结论保留；原始记录归档 |

## 6. 维护规则

1. 新实验先写协议与结论边界，再冻结配置；不把工程检查写成任务收益。
2. 同一主题出现新决策时，更新本页“决策消解”，旧文档移入 `archive/raw/`，不原位改写冻结配置。
3. JSON 收据是实验细节证据，原地保留；移动 md 后必须更新活跃文档中的链接。
4. 归档只移动文件，不删除；删除须另走保留清单与独立审核。
