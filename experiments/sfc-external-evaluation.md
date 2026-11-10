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
