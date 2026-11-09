# First sampled update development diagnostic

Register before evaluating this actor. Use the first successfully completed
actual-vLLM actor v5, without selecting among checkpoints by evaluation outcome.
It used one FSC TRAIN recording, six views,12 responses, one full-batch update
at LR1e-5, seed17, with original SFT initializer. All pair bonuses were zero.

Evaluate its saved unmerged adapter on the unchanged FSC186-view development
manifest and SLURP168-view transfer manifest, using the same original base,
greedy64-token decoding, CPU float32 and seed17 as the completed SFT evaluations.
Reuse those immutable SFT predictions after artifact verification. Compare
updated actor minus SFT, with strict common scoring and2,000 speaker bootstrap
draws, seed17. Report all variants, format/schema validity, false refusals,
non-null correctness and paired both-correct rates; retain negative deltas.

This fixed one-update diagnostic checks whether the executable update changes
heldout decisions. It is not a matched-budget SFT/GRPO/pair-GRPO comparison,
does not isolate pair reward, and cannot establish multi-seed learning or final
generalization. No test inference, checkpoint selection, early stopping or
training-parameter adjustment is driven by these results. The separately
registered continuation chains and mixed-audio integration runs remain fixed.

Use immutable evaluation snapshots and read-only adapter/model/data mounts,
four CPU cores and16GiB per evaluation. Verify the completed actor terminal,
Docker exit and saved adapter hashes before launching. The two existing SFT
comparators must have successful exact container IDs and complete matching
predictions before finalization. Do not alter the historical SFT outputs.
