# Fixed FSC-to-SLURP development comparison v1

Freeze selection before either policy generates outputs. Compare original
Qwen3-ASR-0.6B-hf against the single completed 47-update FSC SFT initializer.
No SLURP fitting or checkpoint selection. Model changes motivated by these
development outputs require a new run and cannot be called untouched transfer.
Test and heldout-test remain unused for model inference.

Select minimum SHA256(source_group) per scenario across official valid and
heldout-valid. Add minimum-hash groups per partition × requested slot type ×
present/null, where available. Deduplicate the union and retain all seven views
of every chosen recording. This intentionally enriches rare non-null slots;
aggregate accuracy is not an estimate of the natural SLURP prevalence.
Builder records each selection bucket and source hashes, verifies complete
views and no audio-hash overlap with FSC train. See the selection receipt in
`docs/engineering/slurp-eval-selection.json` for exact counts and release ID.

Greedy HF generation, FP32, 64 new tokens, seed 17, no answer repair or constrained
decoding. Same prompt contract and strict JSON scorer for both policies.
Preserve raw text, token IDs, generation time, format failures and length limits.
Report scenario/partition/variant/target-presence support and exact accuracy,
choice paired both-correct, schema validity and malformed output counts.
All-null, first-candidate/constant-value and oracle controls precede model scoring.

"Seen" and "heldout" refer to the SLURP release's domain partition, not domains
seen by the FSC-only policy. Both partitions are cross-dataset for this experiment.
Multiple source recordings may share an utterance; do not treat derived decisions
as independent. Report speakers/utterances; official SLURP speakers overlap its
train split. This small development diagnostic is not a final generalization
claim, native intent reproduction, or full official SLU-F1.

Base inference runs independently. SFT inference may start only after the actual
SFT container exits zero and its terminal/result receipts prove all 47 updates.
The saved completed adapter is the only selected checkpoint.

## Frozen release and running handles

Selection `8da2269b1e21e65b20d96100e11ddc092ab6414e433aef437bda7689f0aa5327`
contains 24 source recordings, 24 utterances, 21 speakers and 168 decisions.
Scorer controls: oracle 168/168, all-null 87/168, first-candidate/constant-value
5/168, malformed 0/168. Per-slot presence controls are recorded in
`docs/engineering/slurp-transfer-controls.json`; aggregate all-null accuracy
51.79% illustrates why present-only performance must be separately reported.

On `npu`, container `multimodal-jev-slurp-transfer-eval-base-v1` runs the original
model. Host supervisor PID 2954748 observes the existing SFT container and will
launch `multimodal-jev-slurp-transfer-eval-sft-v1` after proven completion.
Both run roots are under shared `experiments/multimodal-jev/` with the respective
suffix names, copied immutable source/config snapshots and separate outputs.
Observe actual process/container state before resuming; never restart on a stale
progress file alone. No model metrics have completed at registration time.

## Paired comparison (frozen before completed inference)

Use `tools/jev/compare_audio_policies.py` after both runs have exit-zero evaluation
receipts and complete prediction sets. It re-scores raw outputs with one scorer
revision, retaining original run metrics. Verify the manifest/prediction hashes
and equal decoding length, device, dtype and seed. The command takes `--manifest`,
`--base`, `--candidate` (evaluation run directories) and a new `--output` path.

Report candidate-minus-base differences with a paired speaker-cluster bootstrap:
2,000 draws, seed 17, 2.5/97.5 percentile endpoints. Each draw samples speakers
with replacement, including all their eligible recording views for both policies.
The statistic remains the decision-weighted accuracy difference within each
partition/variant/presence stratum. Pair both-correct uses recording pairs as
observations clustered by speaker. A stratum with fewer than two speakers gets
no confidence interval. No aggregate cross-variant success metric is introduced.

These are exploratory development intervals with small cluster counts; they are
not multiple-testing-adjusted significance tests or population guarantees. Shared
utterance text across speakers can introduce dependence not captured by speaker
clustering. Final claims require the full frozen test protocol, multiple training
seeds and appropriate source-family sensitivity analysis.

## Completed v1 development result (2026-11-10)

Both exact Docker IDs exited zero. Complete prediction sets and original hashes
passed verification and common re-scoring. SFT JSON format validity is 168/168,
versus 0/168 for the base. Each Choice view scores 0/24; present slots score
0/9. All 81 non-null targets fail. The SFT model emits 143 nulls and 25 non-null
values; all 71 correct decisions are absent slots. This is below the all-null
control's 87/168, and both paired Choice correctness rates remain 0/24.
Generation totals 2,505.23 CPU seconds (four threads, excluding model loading).
The result supports format learning but no effective semantic transfer on this
selection. See the [receipt](../docs/engineering/slurp-sft-development-v1-result.json)
for full variant/presence counts and partitioned speaker-bootstrap intervals.
This observed development failure does not establish behavior on the full task.
