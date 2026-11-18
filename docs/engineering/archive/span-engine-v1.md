# Span / verl 引擎探索归档（2026-11-10 至 2026-11-12）

本文件收编早期 typed Span、verl/vLLM/TP4/FSDP 工程探针与规则信用实验。原始 md 已移入
`archive/raw/span-engine/`；训练、推理和验收 JSON 收据仍在 `docs/engineering/` 原名保留。

## 做了什么

1. 在 Qwen3-ASR 上接共享概率头，一次前向输出多个 typed Span 概率。
2. 打通 verl actor、vLLM rollout、LoRA 切换、TP4 逐层对齐、四卡 FSDP 与恢复。
3. 在 27 条 learning 录音、72 请求、36 个互补规则对上比较 SFT、联合 GRPO、
   条件信用和成对信用。

## 有效结论

| 主题 | 结果 | 边界 |
| --- | --- | --- |
| HF/vLLM 一致性 | 216 请求验收通过，最大候选 logprob 误差 1.1920928955078125e-7；更新后重载误差 0 | 工程一致性，不是任务收益 |
| 四卡 FSDP | 完成一次真实语音 LoRA 更新；导出重载误差 0；TP1 vLLM 验收通过 | 单步工程验收；不等于完整训练 |
| seed17 三方法 | SFT 123/256=48.05%，GRPO 127/256=49.61%，conditional 126/256=49.22% | source-group 配对 95% 区间均含 0；单 seed、未做多重比较校正 |
| 互补规则 | GRPO/conditional 的三问互补对仅 1/128 两问均正确；常量 false 即可接近 50% | 接近 50% 不代表可用决策能力 |
| 同起点一次规则更新 | RL 与同辅助权重 SFT 约 24 胜/25 负/23 平；互补两问均正确 0/36 | 未观察到稳定 RL 优势 |
| 条件信用单步 | Final 与初始/SFT 持平；相对 SFT/GRPO 的区间跨 0 | 不能归因单个机制；一次更新未体现 PPO 多步作用 |

**结论：工程探针与训推一致性可用；规则信用/条件信用没有证明稳定决策收益，不能把局部点估计或
常量策略的 50% 准确率当作算法胜出。** 后续主线不再用这套小样本规则实验作为发布前提。

## 失败/过期路径

- 用单问 SFT 辅助项的最高点挑选“优胜模型”：未在多 seed 与独立任务上验证。
- 把 GRPO/conditional 接近 50% 解释为规则执行能力：互补对结果否定了这一点。
- 成对奖励、条件信用、TP4 局部对齐等候选：保留为历史诊断，不再直接派发。

## 保留证据

- `span-transfer-three-methods-s17-a2-*.json`、`span-rule-*-comparison-*.json` 等原 JSON。
- 当前仍活跃的接口/运行时文档留在 `docs/engineering/`：
  `single-pass-spans-v1.md`、`span-bf16-runtime-profile-v1.md`、
  `span-fsdp-audio-npu4-v1.md`、`span-resume-v1.md`、
  `verl-vllm-lora-span-alignment-v1.md`、`verl-npu-logprob-source-v1.md`。
