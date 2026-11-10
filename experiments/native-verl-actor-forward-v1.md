# Native verl actor forward gate

Call the unmodified pinned `DataParallelPPOActor._forward_micro_batch` on four
responses for two real TRAIN recordings. Select noul.True from the first group
and native from the second group of seed17's fixed comparison schedule. Use
the original Qwen3-ASR 0.6B HF base, CPU float32, no fused kernels, no packing,
no sequence parallelism and temperature1. The repeated responses are gold/null
fixtures, explicitly not sampled model outputs.

Compare every effective response token to individual unpadded HF forwards.
Fix atol3e-4/rtol1e-5 before execution, as in the prior audio padding probe.
Require finite outputs and exact native actor source SHA256 from the pinned
image audit. Initialize one Gloo rank only to satisfy the constructor's rank
query; do not patch device functions, native forward code or model weights.
The CPU float32 autocast context may warn and disable autocast; retain logs.

This checks a native actor method, not the complete compute_log_prob wrapper,
which calls get_device_id, nor update_policy, Ray workers, FSDP or multi-rank
synchronization. Those remain separate required integration steps. Run on four
remote CPU threads with64GiB and no accelerator devices; preserve immutable
source, exact container and terminal evidence. A failure is retained without
automatic fallback or retry.

## Retained v1 failure and explicit v2 revision

The unmodified v1 container exited1 after the model forward. The pinned verl
`logprobs_from_logits` dispatches by installed kernel availability, not tensor
device, and attempted `npu::npu_cross_entropy_loss` on CPU logits. The
[failure receipt](../docs/engineering/native-verl-actor-forward-v1-failure.json)
retains the exact Docker state and application terminal hash. This is not a
numerical parity failure and does not count as a passed native actor gate.

Register v2 with the same recordings, model, dtype, thresholds and native actor
source. Explicitly opt into `--cpu-logprob-adapter`, replacing only the actor's
imported log-prob dispatcher during the comparison with CPU log_softmax/gather.
The scoped adapter restores the original callable on exit and rejects non-CPU
tensors; it does not edit the installed image or native forward implementation.
Its float64 values and gradients match independent PyTorch cross entropy on
noncontiguous logits within1e-12. Three CPU tests cover these checks, shape
rejection and restoration after exceptions. The v2 result must identify this
adapter; even a pass cannot be described as an unmodified native worker run.
