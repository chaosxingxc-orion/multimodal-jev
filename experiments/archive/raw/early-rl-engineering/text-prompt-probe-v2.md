# Task-specific text prompt probe

Register before inference. The completed SLURP v1 development controls each
produced168 malformed outputs, all with Markdown fences and none length-limited.
Retain those strict scores; do not repair responses or reinterpret them as valid.

Compare the unchanged v1 prompt with v2, which explicitly prohibits Markdown and
extra keys and describes only the active decision type. Both receive identical
schema definitions and reference transcripts. No demonstration answers, target
values or source metadata enter either prompt. The strict parser is unchanged.

Select the lexically first source_group in the existing FSC TRAIN manifest and
the lexically first source_group in the filtered SLURP TRAIN manifest. Include
all their views in fixed example-ID order. Selection is independent of target,
predictions and development results; no heldout SLURP training scenarios enter.
These13 prompts cover Choice, Noul and open slots; they do not cover Score.
Keep source hashes and a selection receipt outside Git with a small copied
receipt in docs/engineering. This is reference-text prompt validation, not
supervised training, ASR robustness or a generalization evaluation.

Use the same Qwen3-0.6B checkpoint/revision, CPU float32/eager, four threads,
non-thinking greedy decoding,64 new tokens and seed17. Separate immutable
snapshots/outputs for v1 and v2, one12GiB CPU container each. Persist all raw
responses and token IDs. Require both containers to exit0 and all13 records to
be present before comparing. Advance v2 to a separately registered development
run only if strict format validity is13/13 and semantic correct count does not
decrease relative to v1. Report schema validity and each task type as well.
Failure is retained; do not search multiple prompts on development predictions.
Passing this small TRAIN gate does not establish a useful text upper bound.

The runtime defaults to v1 and valid-only evaluation. TRAIN input requires
explicit reference-text prompt_probe scope. Text evidence comparisons reject
mismatched prompt versions, so an ASR-versus-reference comparison cannot silently
mix the two protocols. Existing running jobs retain their snapshots.

Both exact containers exited0 and independent rescoring passed. v1 strict
format validity and correctness are0/13; v2 format validity13/13, schema
validity11/13 and correctness4/13. All five unknown targets are answered
incorrectly rather than refused. The registered advance gate passes, with
substantial remaining semantic/schema limitations. See
[result receipt](../../../../docs/engineering/text-prompt-probe-v2-result.json).
