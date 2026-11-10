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
