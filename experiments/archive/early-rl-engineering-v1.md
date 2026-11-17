# 早期 RL / SFT / vLLM 工程实验归档（2026-11-10）

本文件收编首次 SFT/RL 更新、算法预算对照、vLLM CPU 接入、文本证据对照和早期 NPU 诊断。
原始 md 已移入 `archive/raw/early-rl-engineering/`；运行收据仍在 `docs/engineering/` 原路径保留。

## 做了什么

- 首次真实 SLURP / FSC RL 单步更新与 SFT 初始化。
- 三种子 SFT/GRPO/pair-GRPO 训练安排与 token/音频预算核算。
- vLLM ARM CPU rollout、混合长度与 waveform/feature 两条预处理路径。
- ASR 转写与参考转写文本对照。
- 成对优势归一化、reference KL、响应位置投影、NPU sampler/runtime 诊断。

## 有效结论

| 主题 | 结果 | 边界 |
| --- | --- | --- |
| 首次 FSC RL 单步 | native 5/31 → 9/31；Noul balanced accuracy 54.84% → 56.45%；配对全对 3/31 → 4/31 | 单录音、单更新、单 seed；零成对奖励；区间下界含 0 |
| 首次 SLURP RL 单步 | 非空目标 0/81，SFT 与 RL 均无改善 | 同时否定跨任务迁移；只有格式与空输出 |
| vLLM 波形路径 | 最大 log-prob 误差 0.16856，超过 0.02 门槛 | 预处理差一帧；不是训练收益 |
| vLLM 特征路径 v4 | 单条最大误差 0.0000802；完整 actor 批次 0.0000566，均通过 0.02 | 单录音/固定身份；工程一致性，不是效果 |
| 文本证据对照 | ASR WER FSC 14.56%、SLURP 17.90%；参考转写仅小幅改变 Choice，布尔 schema 仍 62/62 违规 | 文本串联只作历史诊断，不再派发 |
| 算法预算核算 | 15 份配置的有效 token/音频帧可复算 | 未建立同预算执行结论；不能据此选学习率或宣称算法胜负 |
| 成对优势 / KL / 投影 | 数值与归一化诊断保留 | 没有任务收益；不能替代真实策略更新 |

**结论：这些实验建立了 SFT→RL、verl、vLLM 和文本对照的早期工程链路，但没有形成
可发布的任务收益；vLLM 一致性通过只说明固定条件下概率对齐。**

## 失败/过期路径

- 用 ASR/参考转写文本对照替代端到端音频主任务。
- 把单步、单录音、单 seed 的正向变化称为 RL 收益或跨任务迁移。
- 在未冻结同预算停止规则前比较 SFT 与 RL。
- 把 vLLM waveform/feature 预处理修复当作训练加速或模型提升。

## 保留证据

- `docs/engineering/` 下 `first-rl-*-result.json`、`vllm-*-result.json`、
  `pair-advantage-*-result.json`、`text-baseline-*-receipt.json` 等原收据。
- 后续 Span/verl 工程结论见 [Span / verl 引擎归档](../../docs/engineering/archive/span-engine-v1.md)。
