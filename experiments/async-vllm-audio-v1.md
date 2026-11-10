# AsyncLLM audio request gate

Register before execution. Use the first TRAIN recording's six schema views
from the completed `vllm-cpu-rollout-sft-v4` configuration, same SFT merged policy,
raw-policy temperature1/top-p1/top-k-1, n2, max32 tokens, seed17, CPU FP32 and
max context2048. Replace the synchronous LLM engine with actual vLLM AsyncLLM.
Keep exact actor feature input and its previously disclosed passthrough adapter.

Requests carry raw prompt text with one audio placeholder, never expanded HF
IDs. Before engine submission check the expanded actor prompt plus max response
fits context; reject instead of silently shortening. At completion require
the engine's expanded prompt IDs to equal HF actor IDs exactly, complete two-
response groups, finite sampled-token log probabilities and valid stop reasons.
Persist random request IDs, policy identity, manifest/audio hashes and budgets.
No requirement of identical RNG outputs across sync/async implementations.

Use the pinned vLLM CPU image on four cores0–3/16GiB and serialize behind the
already registered generic two-step and budgeted-RL gates. Wait for the budgeted
RL supervisor's successful terminal, thereby not racing its later sampler.
An observation timeout is not a restart reason. After exact successful sampler
exit and application terminal, run the existing independent HF per-response
logprob audit at the unchanged .02 threshold. No optimizer update or test/dev
evaluation is part of this gate.

This is the asynchronous engine boundary needed by a future verl audio server.
It does not itself adapt the native Ray manager, server class or agent-loop
postprocessing. Those must preserve all n replicates (native TokenOutput currently
returns one), raw/expanded prompt ownership, features and non-vision position IDs.
The adapter currently serializes request admission and does not establish
multi-request concurrency, cancellation at Ray level, live weight synchronization,
NPU execution or algorithm efficacy. Unit tests cover incomplete streams/groups,
identity mismatch, context rejection and cancelled/failed request aborts.

The [queue receipt](../docs/engineering/async-vllm-audio-v1-queued.json) pins
sourcea3e97d8 and the byte-identical earlier synchronous sampling configuration.
Its supervisor verifies the live budgeted-RL supervisor while waiting, then
requires its successful terminal and exact rollout container exits. No actual
async model inference is claimed at queue creation. Nine request-contract tests
pass without a model; actual AsyncLLM generation and HF parity remain pending.
