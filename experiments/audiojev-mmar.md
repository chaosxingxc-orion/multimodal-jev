# AudioJev released-checkpoint reproduction: MMAR

Scope: evaluate the single published RD-SKL lambda=0.5 seed=20261001 checkpoint
on every one of the 1,000 official MMAR questions. The model-card reference is
56.30%, using the publisher word-set rule. This is not reproduction of training
or the three-seed paper mean of 55.33%. No local model score is available yet.

Use the pinned official AudioJev repository at
`8eeeab5e77d2c410fd6058e7bdf2b4125f11903e`, its unmodified CUDA BF16 SDPA backend,
torch 2.8.x, Transformers 4.57.6 and PEFT 0.17.x. These dependencies belong in a
separate environment; do not replace the NGA-aligned verl/vLLM runtime. An Ascend
or CPU adaptation must be reported separately and numerically validated.

## Input and scoring contract

- Keep all source questions and candidate descriptions in their original order,
  including whitespace and empty padding candidates. Do not synthesize candidates
  from answers. Four questions contain blank candidates; six reference strings
  are not a single candidate. They remain in the full denominator.
- Call `OmniBackend.predict_many` with the official `render_prompt` directly.
  The public `AudioJev.system_one` API sorts unique descriptions and rejects
  blanks; that API is unsuitable for original-order reproduction.
- Save the 2–36 candidate probabilities, selected position, prompt SHA256,
  source/model/code identities and usage for every question. No generated text
  parsing and no JSON-output adaptation of the model.
- References are kept in a separate file and never loaded by the inference
  runner. Full test data are evaluation-only; no tuning, early stopping or
  checkpoint selection on these labels.
- The official `string_match` function is loaded unchanged from the pinned
  publisher source. Missing predictions count as incorrect out of 1,000; the
  original publisher CLI silently skips them. Also report strict selected
  description equality and modality/category breakdowns. Neither metric repairs
  multi-answer references.
- All WAVs must decode completely with finite samples; reject unexpected prompt
  or audio length instead of silently truncating. Reuse original audio files
  across releases; derived files contain absolute paths and content hashes.

## Commands

Build the shared input release using the local `audio-audit` optional dependency:

```bash
.venv/bin/python tools/jev/build_mmar_release.py \
  --root /mnt/cephfs/xingchao/datasets/mmar \
  --output-root /mnt/cephfs/xingchao/datasets/jev/mmar-official
```

On an allocated CUDA device in the isolated official environment, with an
immutable source snapshot and detached process/terminal supervision:

```bash
python tools/jev/infer_audiojev_mmar.py \
  --config configs/experiments/audiojev-mmar.json --output /experiment/run
```

Score only persisted predictions using `tools/jev/evaluate_mmar.py` with
`--release`, `--predictions` and `--publisher-scorer`. A complete inference run
must have terminal exit code 0, result count 1,000 and matching prediction hash.
The scorer can expose incomplete coverage, but that is not a completed baseline.

Remaining gates: isolated official environment, suitable CUDA allocation, actual
full inference, complete terminal receipt and comparison with the model card.
Using the official renderer/backend narrows method differences; it does not by
itself prove byte-identical unpublished training/evaluation prompts or numerical
equivalence to the reported run. Any score discrepancy requires investigation.

## Local validation

Release `041a543cc12bbf7b5bfa9e20e7c4a16ec3ca5b7efa5b2551957a3a02b93124e5`
contains all 1,000 questions and 1,000 fully decoded finite WAVs; longest audio
is 56 seconds. Original question/choices and official renderer output matched
for every item. See the [release receipt](../docs/engineering/mmar-official-v1-receipt.json).

[Scorer controls](../docs/engineering/mmar-controls.json) give 279/1,000 for
always selecting the first candidate under publisher scoring, versus 272/1,000
under exact description equality. Selecting each question's best available
publisher-matching candidate yields 995/1,000. Five references therefore cannot
be satisfied by any single supplied option under that scoring rule. This is a
reference-aware diagnostic upper bound for this output contract, not a model
result. Missing every prediction yields 0/1,000, preserving the denominator.
