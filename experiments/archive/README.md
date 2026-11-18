# experiments 归档索引

`experiments/archive/raw/` 保存历史实验协议的原始文件，按家族组织；本目录的总结只保留
有效结论、失败路径和证据边界。JSON 收据不在归档范围内，仍在 `docs/engineering/` 和
共享实验目录原路径保留。

| 家族 | 原始文件 | 总结 | 内容 |
| --- | ---: | --- | --- |
| `frozen-baselines` | 32 | [冻结基线归档](frozen-baselines-v1.md) | Omni/Qwen3-ASR/FSC/SLURP 冻结推理、提示优化、早期迁移 |
| `early-rl-engineering` | 24 | [早期 RL 工程归档](early-rl-engineering-v1.md) | 首次 SFT/RL、预算对照、vLLM CPU、文本对照、NPU 诊断 |
| `span-rl-probes` | 22 | [Span/BF16 实验归档](span-rl-probes-v1.md) | typed Span、BF16 LoRA、单步 verl、pooling、规则信用 |
| `complex-deferred` | 9 | [复杂任务归档](complex-deferred-v1.md) | SFC、τ² voice、SLURP 时间角色 |

## 使用规则

1. 归档协议只用于追溯，不直接重新派发；新实验须另立预注册协议和独立配置。
2. 归档中的“运行中/等待”只代表记录当时，当前状态必须现场核验。
3. 需要引用历史数字时，引用总结中的边界，并回查原 JSON 收据。
4. 移动文件不改变其证据身份；不要因为归档而把未完成项写成已完成。

## 原始文件清单

- `raw/frozen-baselines/`：`audiojev-mmar.md`、`fsc-audio-sft-init.md`、
  `fsc-frozen-audio-full.md`、`fsc-frozen-audio-pilot.md`、
  `fsc-omni-official-test-v2-{comparison,parent}.md`、`fsc-schema-evaluation.md`、
  `fsc-speechbrain-official-v1.md`、`frozen-*.md`、`slurp-omni-native-test-v1.md`、
  `slurp-jev-full-*.md`、`slurp-transfer-*.md`、`slurp-asr-jev-dp4-lock-v1.md`、
  `pretrained-policy-probe.md`。
- `raw/early-rl-engineering/`：`algorithm-comparison-v1.md`、`first-rl-development.md`、
  `audio-rl-iteration.md`、`budgeted-*.md`、`container-rl-schedule.md`、
  `distributed-verl-loss-v1.md`、`pair-advantage-signal.md`、`reference-kl.md`、
  `response-logits-projection.md`、`sft-continuation.md`、`token-budget-comparison-v1.md`、
  `vllm-*-integration.md`、`text-*.md`、`npu-*.md`。
- `raw/span-rl-probes/`：`native-verl-*.md`、`span-actor-update-v1.md`、
  `slurp-native-spans-*.md`、`slurp-bf16-*.md`、`slurp-vllm-spans-*.md`、
  `slurp-span-*.md`。
- `raw/complex-deferred/`：`sfc-*.md`、`tau2-environment-bridge.md`、
  `slurp-temporal-*.md`。
