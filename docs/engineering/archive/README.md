# 工程文档归档索引

> 本目录收编从 `docs/engineering/` 活跃目录移出的历史设计与失败探索。
> 原件保留在 `raw/<family>/`，不删除；活跃结论见 [`../README.md`](../README.md)。
> 运行/验收 JSON 收据仍按原名在 `docs/engineering/` 保留，移动的只是叙事 md。

| 家族 | 总结 | 归档文件数 | 原始目录 |
| --- | --- | ---: | --- |
| Omni 基线 | [omni-baseline-v1.md](omni-baseline-v1.md) | 35 | [`raw/omni-baseline/`](raw/omni-baseline/) |
| Span / verl 引擎 | [span-engine-v1.md](span-engine-v1.md) | 45 | [`raw/span-engine/`](raw/span-engine/) |
| Intent RL | [intent-rl-v1.md](intent-rl-v1.md) | 110 | [`raw/intent-rl/`](raw/intent-rl/) |
| Context RL 历史候选 | [context-rl-history-v1.md](context-rl-history-v1.md) | 32 | [`raw/context-rl-history/`](raw/context-rl-history/) |
| SLURP 数据/输入历史 | [slurp-data-history-v1.md](slurp-data-history-v1.md) | 14 | [`raw/slurp-data-history/`](raw/slurp-data-history/) |
| 复杂任务 deferred | [complex-tasks-deferred-v1.md](complex-tasks-deferred-v1.md) | 27 | [`raw/complex-tasks-deferred/`](raw/complex-tasks-deferred/) |
| 其他历史 | [misc-history-v1.md](misc-history-v1.md) | 5 | [`raw/misc-history/`](raw/misc-history/) |

## 文件清单

### Omni 基线

- `frozen-fsc-tfrl-progress-20261110.md`
- `frozen-three-stage-progress-2026-11-10.md`
- `mmar-omni-native-v1.md`
- `omni-actor-batch-v1.md`
- `omni-actor-forward-entry-v1.md`
- `omni-actor-source-audit-v1.md`
- `omni-async-boundary-v1.md`
- `omni-buffer-precision-v1.md`
- `omni-compact-keys-diagnostic-v1.md`
- `omni-compact-verl-v1.md`
- `omni-expert-conversion-probe-v1.md`
- `omni-fsdp-forward-storage-v1.md`
- `omni-fsdp-initialization-plan-v1.md`
- `omni-fsdp-layout-v1.md`
- `omni-fsdp-loading-v1.md`
- `omni-intent-reference-coverage-v1.md`
- `omni-npu-layer-probe-v1.md`
- `omni-npu-probability-preparation-v1.md`
- `omni-padding-mechanism-v1.md`
- `omni-pretrained-forward-train6-v1.md`
- `omni-pretrained-layer-diagnostic-v1.md`
- `omni-pretrained-precision-diagnostic-v1.md`
- `omni-pretrained-verl-comparison-v1.md`
- `omni-probability-audit-v1.md`
- `omni-rollout-logprob-contract-v1.md`
- `omni-sampling-acceptance-v1.md`
- `omni-sampling-queue-v1.md`
- `omni-sampling-runner-v1.md`
- `omni-text-position-probe-v1.md`
- `omni-thinker-loading-audit-v1.md`
- `omni-verl-wrapper-probe-v1.md`
- `qwen3-omni-heldout-protocol-v1.md`
- `qwen3-omni-selection-protocol-v1.md`
- `qwen3-omni-slurp-jev-transfer-protocol-v1.md`
- `qwen3-omni-slurp-probe-plan-v1.md`

### Span / verl 引擎

- `audio-rl-integration.md`
- `fsdp-lora-npu4-v1.md`
- `multicard-model-audit-v1.md`
- `rollout-training.md`
- `span-dp4-resume-acceptance-queue-v1.md`
- `span-dp4-resume-drain-queue-v1.md`
- `span-finite-actor-dp4-cpu-v1.md`
- `span-finite-credit-cpu-v1.md`
- `span-finite-resident-dp4-cpu-v1.md`
- `span-finite-session-dp4-cpu-v1.md`
- `span-fsdp-residency-controller-v1.md`
- `span-fsdp-residency-v1.md`
- `span-fsdp-resume-prepared-v1.md`
- `span-k-projection-shape-v1.md`
- `span-lora-switch-v1.md`
- `span-native-actor-cpu-v2.md`
- `span-native-actor-dp4-cpu-v2.md`
- `span-pair-audio-after-sft-queue-v1.md`
- `span-pair-audio-pilot-prepared-v2.md`
- `span-policy-v1.md`
- `span-resident-policy-suite-controller-v1.md`
- `span-resident-policy-suite-v1.md`
- `span-resident-rollout-v1.md`
- `span-resident-suite-host-acceptance-v1.md`
- `span-rule-conditional-comparison-v1.md`
- `span-rule-credit-diagnostic-v1.md`
- `span-rule-matched-comparison-v1.md`
- `span-rule-policy-comparison-v1.md`
- `span-rule-sft-runtime-v2.md`
- `span-selection-initial-sft-v1.md`
- `span-session-offloaded-acceptance-v1.md`
- `span-session-update-consensus-v1.md`
- `span-tp4-complete-k-v1.md`
- `span-tp4-full-width-v1.md`
- `span-tp4-hf-order-suite-v1.md`
- `span-tp4-layer-trace-v1.md`
- `span-tp4-native-suite-v1.md`
- `span-tp4-repeatability-v1.md`
- `span-training-session-evidence-consensus-v2.md`
- `span-transfer-conditional-s17-a2-comparison-v1.md`
- `span-transfer-conditional-s17-a2-v3.md`
- `span-transfer-conditional-s17-failure-analysis-v1.md`
- `span-transfer-conditional-s17-update-diagnostic-v1.md`
- `span-transfer-seed29-launch-recovery-v1.md`
- `span-transfer-three-methods-s17-a2-v1.md`

### Intent RL

- `intent-adam-direction-step23-cpu-v1.md`
- `intent-batch32-accumulation-v1.md`
- `intent-batch32-control-prepared-v1.md`
- `intent-batch32-controller-v1.md`
- `intent-behavior-consumer-v1.md`
- `intent-beta1zero-continuation-v1.md`
- `intent-beta1zero-first-controller-v1.md`
- `intent-binding-first-controller-v1.md`
- `intent-binding-first-step-plan-v1.md`
- `intent-binding-inputs-v1.md`
- `intent-binding-learning-metrics-v1.md`
- `intent-binding-npu-entrypoints-v1.md`
- `intent-binding-resume-contract-v1.md`
- `intent-binding-tensor-transport-v1.md`
- `intent-binding-training-candidate-v1.md`
- `intent-binding-training-gate-v1.md`
- `intent-bounded-credit-scaling-step06-v1.md`
- `intent-candidate-intervention-v1.md`
- `intent-composition-outcomes-v1.md`
- `intent-correctness-replay-v1.md`
- `intent-credit-concentration-all89-v1.md`
- `intent-d12-beta1zero-control-prepared-v1.md`
- `intent-d12-lr2e5-control-prepared-v1.md`
- `intent-d12-lr2e5-full-chain-v1.md`
- `intent-development-overlap-v1.md`
- `intent-dqp-matched-effects-v1.md`
- `intent-finite-continuation-recovery-v1.md`
- `intent-finite-d12-complete-v1.md`
- `intent-finite-d12-pilot-v1.md`
- `intent-finite-d12-step10-v1.md`
- `intent-finite-d12-step11-v1.md`
- `intent-finite-d12-step3-v1.md`
- `intent-finite-d12-step4-v1.md`
- `intent-finite-d12-step5-v1.md`
- `intent-finite-d12-step6-v1.md`
- `intent-finite-d12-step7-v1.md`
- `intent-finite-d12-step8-v1.md`
- `intent-finite-d12-step9-v1.md`
- `intent-finite-learning-curve-v1.md`
- `intent-finite-step1-learning-change-v1.md`
- `intent-finite-step2-learning-change-v1.md`
- `intent-finite-update-queue-v1.md`
- `intent-finite-update-queue-v2.md`
- `intent-initial-learning-diagnostics-v2.md`
- `intent-initialization-development-overlap-v1.md`
- `intent-learning-trajectory-v1.md`
- `intent-matched-process-outcomes-v1.md`
- `intent-objective-integration-v1.md`
- `intent-optimizer-control-v1.md`
- `intent-p-lr-utility-preflight-v1.md`
- `intent-policy-drift-step12-v1.md`
- `intent-population-geometry-step0to4-v1.md`
- `intent-priority-schedule-v2.md`
- `intent-process-credit-step0and11-v1.md`
- `intent-process-outcomes-step0and12-v1.md`
- `intent-q-first-update-v1.md`
- `intent-q-prefix09-curve-v1.md`
- `intent-q-second-update-v1.md`
- `intent-q-third-update-v1.md`
- `intent-q10-learning-v1.md`
- `intent-q11-learning-v1.md`
- `intent-q12-learning-v1.md`
- `intent-q4-learning-v1.md`
- `intent-q5-learning-v1.md`
- `intent-q6-learning-v1.md`
- `intent-q7-learning-v1.md`
- `intent-q7-update-mechanism-v1.md`
- `intent-q8-learning-v1.md`
- `intent-q8-training-batch-v1.md`
- `intent-q9-learning-v1.md`
- `intent-qp-d12-controller-v1.md`
- `intent-qp-matched-prepared-v1.md`
- `intent-relation-credit-step0and12-v1.md`
- `intent-residency-replay-controller-v2.md`
- `intent-residency-replay-prepared-v1.md`
- `intent-resident-behavior-consumer-queue-v1.md`
- `intent-resident-behavior-v1.md`
- `intent-resident-resume-v1.md`
- `intent-resident-two-update-v1.md`
- `intent-resource-archive-handoff-v1.md`
- `intent-resource-listener-handoff-v1.md`
- `intent-resource-observer-handoff-v1.md`
- `intent-resume-native-dp4-cpu-v1.md`
- `intent-runtime-admission-queue-v1.md`
- `intent-saved-supervision-audit-v1.md`
- `intent-schedule-balance-v1.md`
- `intent-shared-gradient-step0-cpu-v1.md`
- `intent-shared-gradient-step0-cpu-v2.md`
- `intent-short-audio-dispatch-recovery-v1.md`
- `intent-step2-resource-recovery-v1.md`
- `intent-step8-learning-mechanism-v1.md`
- `intent-update-metrics-v1.md`
- `intent-update45-mechanism-v1.md`
- `intent-utility-classification-trajectories-v1.md`
- `intent-utility-curves12-controller-v1.md`
- `intent-utility-engineering-pilot-v1.md`
- `intent-utility-report-controller-v1.md`
- `intent-zero-delta-control-v1.md`
- `selection-python312-sum-recovery-v1.md`
- `slurp-instruction-recovery-and-predicates-v1.md`
- `slurp-intent-development-reference-audit-v1.md`
- `slurp-intent-development-review-v1.md`
- `slurp-intent-dqp-processor-cpu-v2.md`
- `slurp-intent-finite-dp4-prepared-v1.md`
- `slurp-intent-finite-dp4-suite-prepared-v1.md`
- `slurp-intent-finite-resume-step2-prepared-v1.md`
- `slurp-intent-finite-resume-step2-v2-observed.md`
- `slurp-intent-listening-review-ui-v1.md`
- `slurp-intent-listening-review-ui-v2.md`
- `slurp-intent-reference-admission-v1.md`

### Context RL 历史候选

- `context-acoustic-numeric-diagnosis-v1.md`
- `context-actual-audio-rl-acceptance-v1.md`
- `context-arithmetic-step256-independent-v1.md`
- `context-attribute-parity-audit-runbook-v1.md`
- `context-backbone-gap-localization-v1.md`
- `context-batch-capacity-and-parity-v1.md`
- `context-batched-owner-candidate-v1.md`
- `context-batched-service-candidate-v1.md`
- `context-checkpoint-policy-export-v1.md`
- `context-epoch2-observed-wall-efficiency-v1.md`
- `context-fixed-image-replication-181-v1.md`
- `context-full-loop-acceptance-v1.md`
- `context-full-training-inputs-v1.md`
- `context-initial-policy-admission-v1.md`
- `context-learning-input-cache-v1.md`
- `context-named-runtime-v1.md`
- `context-native-actor-dp-v1.md`
- `context-native-lora-loading-v1.md`
- `context-online-batch-assembly-v1.md`
- `context-online-device-init-failure-v1.md`
- `context-online-runtime-candidate-v1.md`
- `context-policy-activation-candidate-v1.md`
- `context-pooler-native-token-ids-v1.md`
- `context-reference-cache-v1.md`
- `context-rl-throughput-prefix-v1.md`
- `context-rloo-recovery-readiness-v1.md`
- `context-segmented-attention-candidate-v1.md`
- `context-segmented-worker-candidate-v1.md`
- `context-step256-native-metric-semantics-v1.md`
- `context-token-contract-v1.md`
- `context-training-efficiency-review-20261114.md`
- `context-training-efficiency-update-20261114.md`

### SLURP 数据/输入历史

- `slurp-categorical-full-selection-final-v1.md`
- `slurp-efficiency-review-2026-11-14.md`
- `slurp-jev-full-format-v1.md`
- `slurp-jev-native-inputs-cpu-v1.md`
- `slurp-jev-native-policy-prepared-v1.md`
- `slurp-jev-ontology-audit-v1.md`
- `slurp-native-hybrid-actor-v1.md`
- `slurp-short-audio-padding-v2.md`
- `slurp-simple-final-v1-status.md`
- `slurp-span-balanced-rule-controls-v1.md`
- `slurp-span-rule-controls-v1.md`
- `slurp-span-verl-vllm-chain-v1.md`
- `slurp-throughput-analysis-2026-11-14.md`
- `slurp-true-batch-throughput-v2.md`

### 复杂任务 deferred

- `audio2tool-decode-profile-v1.md`
- `audio2tool-engine-capacity-v1.md`
- `audio2tool-full-inference-v1.md`
- `audio2tool-input-inventory-v2.md`
- `audio2tool-jev-capacity-queue-v1.md`
- `audio2tool-jev-inference-v1.md`
- `audio2tool-jev-v1.md`
- `audio2tool-jev-workload-v1.md`
- `audio2tool-native-inputs-v1.md`
- `audio2tool-release-coverage-v1.md`
- `audio2tool-score-contract-v1.md`
- `audio2tool-tier7-protocol-v1.md`
- `mac-slu-spans-train-v1.md`
- `mac-slu-train-readiness-v1.md`
- `sfc-candidate-context-v1.md`
- `sfc-candidate-execution-v1.md`
- `sfc-jev-complete-queue-v1.md`
- `sfc-jev-protocol-v1.md`
- `sfc-learning-reward-audit-v1.md`
- `sfc-native-full-test-v1.md`
- `sfc-omni-waveform-parity-v1.md`
- `sfc-prompt-selection-v1.md`
- `sfc-reward-balance-numerical-v1.md`
- `sfc-sampling-reward-signal-v1.md`
- `sfc-teacher-feedback-readiness-v1.md`
- `sfc-weighted-policy-loss-v1.md`
- `slurp-temporal-pilot-v1.md`

### 其他历史

- `actor-artifact-audit.md`
- `end-to-end-readout-status-2026-11-10.md`
- `external-audio-overlap-v1.md`
- `fsc-context-transfer-v1.md`
- `fsc-jev-slurp-progress-20261110.md`

