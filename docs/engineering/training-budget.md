# Training budget accounting

SFT and actor updates now report effective prompt tokens, response tokens,
valid audio frames and sequence exposures. Prompt counts include expanded audio
placeholders, and response counts include the assistant end token where present.
Padding contributes to none of these effective counts. Repeated schema views
and rollout replicates are exposures, not independent recordings.

The ledger separately counts sequences/tokens processed in gradient-enabled
forwards, no-gradient forwards, and actual optimizer steps. Actor parity checks
count even if all advantages are zero; a frozen-reference forward counts too.
Zero gradients record the gradient work with zero optimizer steps. SFT counts
each teacher-forced example and records one optimizer step per accumulation
group. Its per-step ledgers are saved in the existing resume checkpoint/history.

These are input accounting fields, not FLOPs or wall-clock estimates. They omit
vLLM generation and export/reload checks, which require separate receipts.
Transformer padding, audio encoder work, generation KV-cache reuse and hardware
all affect real cost. Equal updates or equal effective tokens alone cannot
establish equal compute. No existing experiment is relabeled budget-matched.

The new fields apply to new immutable source snapshots. Currently running jobs
retain their registered code and do not acquire instrumentation retrospectively.
The real-processor/verl smoke independently compares the padded batch ledger
against unbatched processor lengths; its response/gradient fixtures are
controlled inputs and do not represent actual policy trajectories or actor steps.

The real verl smoke exited 0 in the NGA-aligned image. Padded batch counts match
independent unbatched processor calls for four fixture responses, including two
different audio lengths. See [receipt](training-budget-v1-result.json). Its
gradient-pass/optimizer-step inputs describe a hypothetical actor budget and
are explicitly labeled budget_fixture; this probe does not load or update model
weights. Full SFT/actor ledger persistence still requires subsequent real runs.

## Generation accounting

New sampler snapshots persist engine-returned prompt lengths, all generated
response lengths, length-limit termination counts and monotonic engine.generate
duration. Engine construction time is recorded separately. Request prompt tokens
count each prompt once; replicated prompt tokens count it once per response.
Neither number estimates actual prefill computation: KV/prefix caching, shared
prompt work, audio preprocessing and hardware affect cost. Do not add these
counts to actor tokens and call the sum FLOPs. Missing engine prompt IDs cause
accounting to fail rather than silently substitute tokenizer estimates or zero.

Earlier rollout receipts do not acquire this instrumentation retrospectively.
The response token count can be recovered from saved records, but engine prompt
lengths and timings cannot be claimed without observed evidence. Unit checks
cover unequal prompt/response lengths, truncation, missing observations and
clearing stale accounting after a failed call; real engine verification remains
required before using the new fields in a comparison.

Register a real-engine accounting probe: the native views of the same two
previously selected mixed-length TRAIN recordings, original SFT initializer,
seed29, n2, unchanged raw-policy decoding and actor_features. Run in the pinned
official CPU vLLM image with four cores and 16GiB. Require four complete
responses; independently sum saved response lengths and compare them with the
ledger. Require two positive engine prompt lengths and finite positive engine
load/generate times. Preserve the outputs as accounting evidence, with no actor
launch or efficacy claim. Root: vllm-generation-budget-v1 in the shared experiment
directory. This checks actual API fields rather than replaying test doubles.
