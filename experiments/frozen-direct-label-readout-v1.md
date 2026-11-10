# Frozen direct-audio single-token readout

2026-11-10. A distinct decoder baseline, not a repair of failed JSON generation.
The same frozen Qwen3-ASR-0.6B-hf directly receives audio plus task alternatives.
One forward pass scores single-token labels at the final assistant-prefix
position. Normalize over supplied alternative labels; map the selected label
to its declared typed value. No transcription, second model, new head or training.

Use original candidate order; labels0–9,A–Z correspond to each supplied choice,
with a null alternative appended only if allow_unknown=true. Noul has false,
true, then allowed null. Label token IDs must be distinct single reversible
tokens. Open slots/full native SLURP are unsupported by this finite readout.
Do not claim constrained format validity as learned JSON-following ability,
or label-normalized probabilities as calibrated confidence. No posthoc repair.

First probe: original locked64 TRAIN FSC recording groups, all384 six-view
requests; no label-driven selection. Fixed prompt direct-label-readout-v1.txt,
HF eager BF16, seed17, context4096, image52ba94e1, physical175 card4.
Run both actual waveform and a same-duration zero waveform control using the
same task definitions. Zero waveform is a sensitivity diagnostic, not a
semantic unknown benchmark; keep its scores separate.
Each condition budget3600seconds, outer3900. Check checkpoint hashes before
and after; preserve all per-label logits, mapped values, prompt hashes and IDs.

Report six-view correctness, native/permutation/rebind/missing both-correct,
Noul both-correct and all-six-correct, plus actual-vs-zero output differences.
Do not count views as independent recordings. No teacher or training dispatch
under this protocol. Only a real, discriminating task signal can justify a
subsequent direct-audio teacher prompt loop.
