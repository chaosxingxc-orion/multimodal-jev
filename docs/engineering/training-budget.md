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
