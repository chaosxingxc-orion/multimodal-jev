# NPU audio runtime admission v1

Use only user-allocated physical Ascend devices 4–7 on SSH host `npu` (175).
Use immutable image `sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`;
preserve its driver/toolkit entrypoint. Mount only these device nodes and required
manager/shared-memory/communication nodes, with no privileged mode. Models,
datasets and the source snapshot are read-only. Network is disabled.

Before model use, require exactly `/dev/davinci4` through `davinci7` and four
logical devices. Run an exactly representable FP32 matmul on every visible
device. Logical device 0 then runs the official Qwen3-ASR-0.6B-hf weights at
revision `7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c`, BF16, eager attention,
with seed 17 and the same registered TRAIN recording as the earlier CPU probe.
This checks container device admission, not communication collectives.

The real-model test includes greedy ASR (32 output tokens), response-only masked
CE with a boolean Jev target, finite response log probabilities, agreement of
CE and negative mean response log probability (atol/rtol 1e-4), backward with
finite positive gradient norm, clipping at 1.0 and one AdamW update. LoRA uses
language-model q/v projections only, rank 4, alpha 8, zero dropout; LR 1e-4,
zero weight decay. Require changed parameters and every optimizer step equal
to 1. Save results and actual Docker/application exit status; retain failures.

This is a hardware/runtime gate. It does not establish semantic accuracy,
native verl actor updates, vLLM generation, distributed model training, or an
academic baseline reproduction. No output checkpoint is used for evaluation.
Existing CPU jobs continue unchanged. Subsequent NPU vLLM and actor parity
gates need their own protocols and receipts.

## Result

Container `41fa0d216d28ece79f55876f5d19e08bcd13dd8ba70ad143d9e6100a8e67fe90`
and the application both exited 0. Four-device arithmetic passed. Single-model
BF16 loss was 10.96512985, CE/log-prob error 9.54e-7, pre-clip gradient norm
42.07757; 56 of 112 adapter tensors changed and all 112 optimizer states reached
step 1. This is consistent with zero-initialized LoRA B matrices causing only
one factor per pair to change at the first step. No accuracy conclusion follows.
Full evidence: [result receipt](../docs/engineering/npu-audio-runtime-v1-result.json).
