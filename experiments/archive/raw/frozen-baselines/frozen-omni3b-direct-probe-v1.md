# Independent frozen Omni 3B direct task probe

**CANCELLED by user before actor dispatch. Reuse existing Qwen3-Omni instead.**
Download and waiting supervisor stopped; no inference result exists.

Motivation: frozen Qwen3-ASR direct JSON has failed on FSC and SLURP TRAIN
probes; finite-label readout also lacks paired discrimination. Preserve all
ASR results. Add the original official Qwen2.5-Omni-3B as a separate frozen
instruction-capable audio model, not a replacement ASR version, trained
AudioJev checkpoint, or gain attributed to ASR prompt optimization.

Pin official revision f75b40e3da2003cdd6e1829b1f420ca70797c34e. Download all
18 files with official LFS SHA256/Git blob checks and retain publisher license.
Use existing NPU image sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c,
Transformers5.14.1, eager BF16, seed17, physical175 card4. This is our NPU
adaptation, not AudioJev's CUDA/Transformers4.57.6 official reproduction.
Only official Thinker weights are used for audio-to-text task inference;
validate every Thinker tensor and allow only official talker/token2wav tensors
to remain unused. No transcript, adapter, head training or weight updates.

First probe: first previously fixed FSC TRAIN learning recording group, all
six Jev views (no label-based choice of example). Same generic frozen decision
instruction and dynamic schema as the ASR JSON interface. Put waveform and
task definition in user content, neutral system instruction; normal assistant
start, no ASR prefix. Greedy free JSON generation, max128new tokens, context4096,
3600s application/3900s outer timeout. No output repair or forced labels.
Do not import the ASR teacher's selected prompt into this distinct interface.

Require complete six predictions, exact audio/model/request identity, zero
weight updates and successful application/container terminal. Score strict
per-view correctness, four paired correctness metrics and all-six correctness;
report malformed/truncated outputs. A single group is a runtime/interface probe,
not task efficacy. If runtime succeeds, fixed64 learning groups and equal-length
zero-waveform control are the next gate before teacher work or full evaluation.
Do not tune on current ASR TEST results; no new remote teacher call scheduled.
