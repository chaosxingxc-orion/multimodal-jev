# FSC frozen baseline and external-teacher prompt experiment v1

**Route correction:** ASR→text teacher/actor experiments below are retired by
the user's explicit end-to-end requirement. Keep their results and costs as
historical diagnostics; p1 is not an end-to-end improvement and its pending
full test was cancelled. New learning feedback must come from an actor that
directly consumes audio and task definitions.

2026-11-10. Implements ../../../../proposal/archive/raw/02-frozen-baselines-tfrl-jev.md without weight training.

Base checkpoints remain the official 0.6B HF ASR and Qwen3-0.6B text models.
HF eager BF16 on allocated 175 NPUs; one process/device; pinned existing image.
The initial instruction is configs/tasks/frozen-decision-prompt-v1.txt, greedy
decoding with max128, seed17, context4096. No adapters, output repair or fallback.
Report invalid JSON as incorrect. Native task is the TRAIN-defined 31 FSC
action/object/location triples, represented by stable candidate IDs. This is
the native joint-intent task through a generative label interface, not the
SpeechBrain published serialization or an existing learned Jev readout head.

The original 64 TRAIN groups are selected by SHA256(source_group) rank. Their
fixed greedy results are engineering feedback, not test scores. Original
test3793 runs in both audio-direct and ASR-text routes without filtering or
retuning; retain their full coverage, raw text and termination evidence.

First external-teacher condition: ASR-text only, same64 TRAIN groups, four
independent draws each at temperature0.7/top_p1/top_k0. Only temperature and
replicate seeds differ from evaluation. Preserve all256 draws/costs. Teacher
material includes the parent instruction, complete group summary, bounded
genuine mixed-success groups and their exact schema, ASR evidence and outputs.
Gold labels are explicitly offline learning diagnostics. Same-output and
same-reward groups are not evidence of a semantic advantage. Do not claim the
text teacher listened to audio. First request proposes at most4 prompts, only
procedural changes; no task facts, per-ID answer mappings or scorer edits.
This is an external-teacher TF-GRPO-inspired prompt loop, not a reproduction
of the full original training-free GRPO algorithm.

Selection population: next256 TRAIN groups in the same SHA256 order, excluding
all64 learning groups; record group IDs before teacher dispatch. Each candidate
and the untouched baseline sees the same ASR evidence, same candidates, greedy
decoding and max128. Keep all candidates and costs. Select highest joint accuracy;
ties retain the parent (among novel tied candidates prefer shorter then ID).
First round capped at1 teacher request/8192 generated tokens and4 candidates.
No second round without a new budget. Test predictions, labels and metrics never
enter teacher material or candidate selection. Test once after prompt selection.

## Registered transport correction v2 (2026-11-10)

The first request terminated with HTTP 400 from the provider's internal MaaS
component; usage is unknown and no prompt candidate was returned. Preserve its
request and terminal receipt. Register exactly one additional request in a new
directory, with identical messages/model/token limit (8192), changing only
`thinking.type` from `disabled` to `enabled`, a mode used in the Omni reference.
This tests compatibility; the generic error does not establish its root cause.
Maximum cumulative requested output is 16384 tokens across these two attempts;
actual usage must be reported separately, including unknown first-call usage.
No automatic retries, repair calls, or candidate promotion. Keep the original
four-candidate limit and all frozen learning/selection/test boundaries. A
truncated or malformed response is retained and rejected without repair.

## Registered completion-budget extension v3 (2026-11-10)

The compatibility attempt received HTTP 200 but terminated with finish_reason
length: 11886 prompt tokens, 8192 completion tokens (8191 reported reasoning),
zero final-content characters. It produced no importable candidate. Preserve
this attempt; it does not establish an optimization result.
Register one further request with the same messages/model/enabled thinking,
max_tokens 32768 and total timeout 900 seconds, in a new immutable directory.
This is a bounded completion-budget extension, not a silent retry or repair.
Cumulative requested output ceiling across all three attempts is 49152 tokens;
the first attempt's unknown usage remains unknown. Keep at most four candidates
and unchanged learning/selection/test populations. No further requests are
scheduled under this protocol; reject another truncated response.

Runtime budgets: full3793 baseline per route max4h; learning256 draws max1h;
selection per prompt max1h. Guard outer runtime has extra30s shutdown grace.
Source is immutable and model files are rehashed before/after. Source groups,
model/prompt/input/result hashes, exact container IDs and application terminal
are retained. Partial/failed runs are not complete benchmark results.

fsc-jev six-view release additionally contains permutation, rebind,
missing-candidate and Noul-pair tasks. Source recording groups inherit splits
and learning/selection role across all views. A later separate prompt loop
uses the same controls and paired both-correct metrics. Neither new labels nor
independent extra recordings are claimed. Other dataset families follow with
native schemas, complete original slots and source-group held-out transfer.
