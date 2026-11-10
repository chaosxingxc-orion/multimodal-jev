# Frozen local Qwen3-Omni direct-audio vLLM probe

Use existing official Qwen/Qwen3-Omni-30B-A3B-Instruct weights at
/mnt/cephfs/xingchao/models/qwen3-omni-30b-a3b-instruct. All publisher files
match revision26291f793822fb6be9555850f06dfe95f2d7e695, verified independently
against official metadata. No model download. Qwen2.5-Omni probe is cancelled.

Image sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c,
existing project verl/vLLM0.30 stack; physical175 cards4,5, TP2. Wait for exact
ASR TEST containers to finish and inspect card mappings before dispatch. No
0–3 use, no restart of another service. Eager BF16, seed17, maxcontext4096,
maxseqs1, maxbatchedtokens4096, GPU memory0.85, CPU16/memory128GiB/shm8GiB.
Application engine init and inference protected by external3600s timeout.

First immutable FSC TRAIN learning source group, all six views; original
frozen-decision-prompt-v1.txt with dynamic task definition. Same processor
chat template, neutral system, user contains actual waveform + task; no
ASR prefix or transcript. vLLM receives raw rendered prompt with waveform,
never pre-expanded HF token IDs. Greedy generated JSON, max128tokens; no
repair, forced labels or second online model. Instruct only, not Thinking.

No teacher calls or training. Require full coverage, request/audio/model hash
custody, unchanged weights and successful app/container terminal before score.
Report invalid/truncated output and all six/paired metrics. One group only
establishes a runtime/interface probe. Subsequent64TRAIN actual/zero controls
need a new registered launch; do not call first-group results a benchmark or
cross-task generalization. This is our NPU/vLLM adaptation, not the publisher's
historical CUDA runtime or HF/vLLM numerical parity validation.
