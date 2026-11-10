# SFC external evaluation protocol v1

SFC-Bench is an external function-calling transfer benchmark for the FSC/SLURP
trained policies. Preserve official `train_id`, `test_id`, `test_ood` and every
level. The published train split is retained as a reference, not mixed into
current Jev training. No test inference or model selection has been performed.
Any later SFC adaptation requires a new protocol and independent holdout.

Source: QwenAudio/FunResearch, revision
`5cd9dc679584ba9b14454908a0f2c85e74c00147`, `SpokenFC`, release `1228`.
Original materials are CC-BY-NC-4.0; retain upstream licenses. Official code is
loaded from the shared dataset checkout, not copied or modified in this project.

## Frozen input contract

`tools/jev/build_sfc_release.py` writes content-addressed JSONL under shared
`datasets/jev/sfc-official/`; see `docs/engineering/sfc-official-v1-receipt.json`.
Each record separates `input` (official messages and ordered audio paths),
`supervision` (unchanged official solution plus parsed calls), and metadata.
Targets and reference transcripts never enter the model-input mapping.
The release has 5,534 / 2,590 / 771 turn records and 4,499 / 1,986 / 593
source conversations respectively. All multi-turn conversations retain both stages.

Stage 2 contains two recordings and supplied assistant clarification history.
It is **offline supplied-history evaluation**, not a closed-loop agent result.
Do not collapse it to the single-audio FSC schema, discard stage 1, or replace
the supplied history with generated text while claiming the same benchmark.
All speech is TTS. The original release provides no speaker identity, so no
speaker-disjoint claim is supported. Audio hashes are disjoint across splits;
train and OOD target-function sets are disjoint. Observed target function counts
are 217 / 216 / 61, distinct from the benchmark's full declared schema inventory.

`NAN` remains the literal string sentinel for missing arguments. Arrays stay
arrays; they are not flattened or discarded. Multi-function outputs retain call
multiplicity. No external tools are actually executed by this benchmark adapter.

## Scoring and reproducibility

Install the optional `sfc-eval` dependencies and run:

```bash
.venv/bin/python tools/jev/evaluate_sfc.py \
  --manifest /path/to/release/test_ood.jsonl \
  --predictions /path/to/raw-predictions.jsonl \
  --official-scorer /mnt/cephfs/xingchao/datasets/sfc-bench/project/eval/metrics_multilevel.py \
  --output /path/to/new-score-directory
```

Prediction records require `example_id` and unchanged `raw_text`. Unknown and
duplicate IDs fail; absent predictions remain failures in the full denominator.
Score ID and OOD separately. The official scorer hash is checked before running
the unmodified CLI. Preserve `official-metrics.txt` and detailed official outputs.
Alongside those numbers, report strict JSON format rate, call-multiset exact
accuracy, function-multiset accuracy, function-bound slot F1, and both-turn exact
accuracy for L3_1 and L3_2. Strict exact is order-insensitive but preserves duplicate
calls, literal values/types, arrays and function/argument binding; it is not an
execution-order metric. Confidence intervals must cluster by source conversation.

The [audit](../docs/engineering/sfc-scorer-audit.json) found these official scorer
limitations using explicit fixtures, without model predictions:

- Array value changes can receive perfect overall accuracy because argument
  extraction stops at non-double-quoted literals. On actual reference labels,
  1,196 / 446 / 8 arguments are omitted in train / ID / OOD parsing, including
  arguments after an unsupported value.
- Swapping values between different functions can retain perfect slot F1.
- Duplicating a function call can retain perfect overall accuracy (set comparison).
- Malformed output versus an empty reference can receive perfect overall accuracy.

Do not silently replace the official metric with our strict one or describe them
as interchangeable. Oracle controls on train validate both paths, but oracle
passing alone does not reveal the above defects. Preserve adversarial fixtures.

## Remaining reproduction gates

The official SFC stack uses Qwen2.5-Omni-7B, ms-swift/LLaMA-Factory and older
vLLM. The paper's final GRPO uses FP16 and 32 generations; checked-in scripts
default to BF16 and 8. Our verl/vLLM port must explicitly document such differences.
No published SFC checkpoint has been loaded or inferred here. The current
Qwen3-ASR GRPO path handles one audio per example. A separate chronological
[multi-audio inference adapter](sfc-audio-adapter.md) is implemented. The
[CPU engine check](../docs/engineering/sfc-vllm-cpu-engine-v3-result.json)
completed one registered two-audio TRAIN request, but the base ASR emitted
transcription rather than valid tool calls. It establishes request acceptance,
not that both audio clips causally influenced the output. Schema-to-call
composition, official-model runs and actual transfer scores remain. Stateful
tool execution is a separate [τ² integration](tau2-environment-bridge.md),
not an alternative scoring path for this offline SFC benchmark.

## Local Qwen3-Omni input mapping

`src/multimodal_jev/sfc_omni.py` now maps the input-only record into native
Omni messages. It follows the pinned official `eval/infer.py` role assignment:
first user text becomes system text, each audio is attached to its chronological
user turn, later user text and supplied assistant history remain unchanged.
The literal `<audio>` marker inside official text is preserved exactly as in
that client; it is distinct from the model's actual multimodal placeholder.
No `tools=` argument is supplied: that would inject the model's XML tool-call
protocol and change the benchmark's required JSON list of call strings.

Full static rendering with the local Qwen3-Omni template passed on all
5534/2590/771 TRAIN/ID/OOD inputs, including 1035/604/178 two-audio turns.
See `docs/engineering/sfc-qwen3-omni-static-input-audit-v1.json` for input and
template hashes. Twelve adapter tests check chronology, history/schema changes,
audio reordering, single-turn retention and rejection of extra supervision.
The audit does not use gold answers to adjust prompts and produces no scores.

This is a Qwen3-Omni adaptation, not a reproduction of the paper's
Qwen2.5-Omni-7B model. Before dispatch, use TRAIN-only engineering inputs to
validate actual processor features, multi-audio vLLM limits and complete context
lengths. Preserve all levels and both turns; never drop an oversized example
or silently truncate it. No new model download or SFC inference has been
launched by this adapter work. Native SFC output remains the official call-list
format; typed SFC-Jev decomposition is a separate future dataset protocol.

The subsequent TRAIN processor check has now completed successfully in the
pinned image, CPU only, without model weights. The first source conversation
lexicographically in each of the four levels yields six complete turns;
expanded inputs contain 577–1044 tokens, all fitting 4096 with a 1024-token
output budget. Both two-audio turns pass the waveform reversal control:
feature rows and masks reverse with the audio, while unreversed features differ.
Maximum reversal residual is 1.1920929e-7. Evidence:
`docs/engineering/sfc-qwen3-omni-processor-train-v1-result.json`.
This verifies feature ordering, not that a model uses both recordings correctly.

A full 8895-input CPU processor audit is running separately under
`sfc-qwen3-omni-full-context-v1`, with no NPU devices mapped. It records actual
expanded context lengths, audio duration limits and every overflow instead of
dropping or truncating samples. Its launch is recorded in
`docs/engineering/sfc-qwen3-omni-full-context-v1-launch.json`; complete results
are still pending. Native vLLM multi-audio generation remains unverified.

The six-turn vLLM TRAIN probe is now prepared at
`/mnt/cephfs/xingchao/experiments/multimodal-jev/sfc-qwen3-omni-train6-v1`,
with frozen source `556b8eb`, original model, BF16/TP2/seed17, context4096,
greedy1024 output tokens, max_num_seqs1, prefix cache disabled and audio limit2.
Selection and messages match the CPU preflight exactly. The inference entry
compares each actual vLLM expanded prompt-token sequence against HF processing
of the same waveforms; a mismatch fails the engineering gate and preserves the
attempt. It hashes model files before/after and keeps raw output and truncation.
Prompt-token equality alone does not prove feature equality or causal use of audio.

The prepared command and mounts target NPU6/7 after the native SLURP TEST and
its queued transfer TRAIN validation. It has not been dispatched; current
device reservations take precedence. Receipt:
`docs/engineering/sfc-qwen3-omni-train6-v1-prepared.json`.
Twenty-two input/runtime tests pass; actual engine generation is still pending.
After successful execution, score precisely these six TRAIN turns using the
existing SFC official-plus-strict scorer, separately from full ID/OOD results.
Do not use the generic FSC/SLURP watcher, which selects a different scorer.

## Completed HF context audit and long-audio correction

The complete CPU audit has finished. Its conservative `n_samples` threshold
initially excluded eight TRAIN requests, while all ID/OOD inputs were processed.
Inspecting the exact image's processor source showed Omni sets `truncation=False`;
the default Whisper 30-second length is not an Omni model duration limit.
Those eight requests were independently processed in a preserved supplement,
including the longest 39.65075-second recording. All frame masks match full
untruncated waveforms and all requests fit the registered context budget.
The first supplement failed because our frame-count check omitted Whisper's
final STFT-frame removal; its failure is retained and a corrected verifier
with boundary tests completed in a new container.

Combined evidence: `docs/engineering/sfc-qwen3-omni-complete-context-v1.json`.
All 8895 inputs now have actual HF context measurements: maximum input lengths
1486/1378/1020 tokens for TRAIN/ID/OOD, plus1024 output fits4096 throughout.
This does not prove the actual vLLM multimodal engine handles them equivalently.
The prepared six-turn probe contains no >30-second audio; its conservative
guard must not be reused as a full-data exclusion rule. A future full runner
must preserve long audio and verify complete feature coverage, rather than
dropping it based on `feature_extractor.n_samples`.

## Queued six-turn engine validation

`watch_sfc_omni_probe.py` is now running from immutable coordinator snapshot
`f1fbc6b`, waiting on the exact native SLURP TEST and transfer TRAIN supervisors.
It binds process start times, launch/configuration hashes and successful
container identities, independently accepts transfer TRAIN execution, checks
that NPU6/7 are free, then dispatches the already prepared inference snapshot
`556b8eb` once. Ambiguous launch or any failure is retained without replay.
Queue receipt: `docs/engineering/sfc-qwen3-omni-train6-v1-queued.json`.

After all six raw outputs succeed, acceptance checks complete IDs, original
input/audio bindings, model/runtime/source identity, token parity and truncation.
Only then are the six original TRAIN labels selected for official-plus-strict
scoring. Invalid JSON is scored as generated; no accuracy gate is imposed.
The exact six-turn scoring integration passed with oracle predictions, including
both-turn counts for L3_1/L3_2; this is a scorer fixture, not model performance.
See `docs/engineering/sfc-qwen3-omni-train6-score-fixture-v1.json`.
Twenty-six related tests passed. The waiting supervisor is not evidence that
SFC vLLM inference or full ID/OOD evaluation has completed.

## Separate native requests and scoring targets

Complete native SFC consumption release:
`/mnt/cephfs/xingchao/datasets/sfc-native/8bc5c8bfc4a6d7ec17197134ac9564daf9f6e7d2770ed854ccf1fae58bd82067`.
Each official split has `*.requests.jsonl` with only ID, split, source group,
original messages/ordered audio paths and audio hashes. Separate
`*.targets.jsonl` files preserve the original solution, parsed calls, metadata
and a SHA256 binding to the request. Runtime messages use only `input`.
This separation excludes the current answer, but deliberately retains the
official supplied assistant history. It is not a claim that prior-turn text
is free of semantic information.

All 5534/2590/771 turns and 4499/1986/593 conversations are preserved.
An independent full field-by-field check reconstructs every source record and
confirms identical Omni message mapping; arrays, duplicate calls and literal
`NAN` survive unchanged. Audio hashes inherit the already verified source;
export additionally checks cross-split hash isolation without rehashing waves.
Receipts: `docs/engineering/sfc-native-consumption-v1.json` and
`docs/engineering/sfc-native-consumption-v1-independent.json`.

The first export attempt rejected stage-specific first-message differences;
no release was published and its temporary output was removed. Inspection
found 918/544/148 two-stage groups with differing first messages, including
function-list order changes, while their audio prefixes and hashes agree.
The corrected exporter preserves each stage's exact original message and
counts the differences rather than enforcing equality or normalizing order.
Twenty-one relevant tests pass. The prepared six-turn probe keeps its original
snapshot and input contract; this release is for subsequent full native runs.
Typed SFC-Jev decomposition and full ID/OOD inference are still outstanding.
