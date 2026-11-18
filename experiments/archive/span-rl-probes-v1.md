# Span / BF16 / 规则信用实验归档（2026-11-11 至 2026-11-13）

本文件收编 typed Span、BF16 LoRA、verl/vLLM 单步生命周期、vLLM pooling 与规则信用实验。
原始 md 已移入 `archive/raw/span-rl-probes/`；训练、推理和验收 JSON 收据仍在
`docs/engineering/` 原路径保留。

## 做了什么

- Qwen3-ASR + 共享概率头的多 Span 读出与 SFT。
- BF16 LoRA 执行、FP32 master/Adam、保存重载与 HF/vLLM 跨后端对齐。
- 单步 verl 更新、导出重载、新版本 rollout 的完整生命周期。
- vLLM 原生多 Span pooling、未合并 LoRA 诊断。
- 27 条 learning 录音、72 请求、36 个互补规则对上的 SFT/GRPO/条件信用/成对信用对照。

## 有效结论

| 主题 | 结果 | 边界 |
| --- | --- | --- |
| HF/vLLM 一致性 | 216 请求通过，最大候选 logprob 误差 1.1920928955078125e-7；更新后重载误差 0 | 工程一致性，不是任务收益 |
| 四卡 FSDP | 完成一次真实语音 LoRA 更新；导出重载误差 0；TP1 vLLM 验收通过 | 单步工程验收；不等于完整训练 |
| 单步生命周期 | step-v4 更新前概率误差 0、gradient norm 3.984375、重载误差 0、更新后 vLLM 误差 ≤1.49e-8 | 单步；未证明多步稳定性或 held-out 收益 |
| BF16 重载 | 20 次更新完成；48 条候选 logprob 最大差 0 | 仅 HF 保存重载一致 |
| BF16 跨后端 | 首次 max logprob 差 0.1405742168，完整动作比率 [0.8115,1.1993]，gate 失败 | 训练 adapter 不同，不能单独归因 BF16 |
| seed17 三方法 | SFT 123/256=48.05%，GRPO 127/256=49.61%，conditional 126/256=49.22% | source-group 配对 95% 区间均含 0；单 seed |
| 互补规则 | GRPO/conditional 三问互补对仅 1/128 两问均正确；常量 false 即可接近 50% | 接近 50% 不代表可用决策能力 |
| 同起点规则更新 | RL 与同辅助权重 SFT 约 24 胜/25 负/23 平；互补两问均正确 0/36 | 未观察到稳定 RL 优势 |

**结论：Span/BF16/verl 工程链路和单步生命周期可复现；规则信用与条件信用没有证明稳定
决策收益，不能把局部点估计或常量策略的 50% 准确率当作算法胜出。**

## 失败/过期路径

- 用单问 SFT 辅助项的最高点挑选“优胜模型”。
- 把 GRPO/conditional 接近 50% 解释为规则执行能力；互补对结果否定了这一点。
- 把单步更新、导出重载或跨后端一致性当作完整训练效果。
- 直接沿用早期多 Span / 规则信用配置作为当前简单成员判决主线。

## 保留证据

- `span-transfer-three-methods-s17-a2-*.json`、`span-rule-*-comparison-*.json`、
  `slurp-bf16-*-result.json` 等原收据。
- 工程侧完整总结见 [Span / verl 引擎归档](../../docs/engineering/archive/span-engine-v1.md)。
- 当前 runtime 接口仍以 `docs/engineering/` 的
  `single-pass-spans-v1.md`、`span-bf16-runtime-profile-v1.md`、
  `span-fsdp-audio-npu4-v1.md`、`span-resume-v1.md` 为准。
