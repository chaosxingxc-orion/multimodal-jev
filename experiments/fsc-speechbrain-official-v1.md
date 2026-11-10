# FSC published SpeechBrain weights: native test retest v1

Reference: published 99.60% FSC test accuracy, not a local result. Evaluate all
3,793 official test utterances with the released SLU checkpoint
`9163a11ee29a14241356a7cc2cc05672b73ba5fe` and recursively required ASR checkpoint
`979a53a7a3f6c9291c02c040fd8ebfb2471cf8a3`. Source is official SpeechBrain
`c0241278f93ef02cd5191f03caf4e5f8e1292bf6`, image
`sha256:696e6adc27f7d1bfcc01633233fd75c7577778a6651be24173c9d66a5d4c0ef0`.
This is a published-weight retest with current official inference, not a
reproduction of the 2021 training environment or a verl implementation of CRDNN.

CPU float32, four threads, seed 17, batch one, unmodified released decoding:
beam 80, temperature 1.25, eos threshold 1.5, native 16 kHz mono waveforms.
Both checkpoints and all inputs are local/read-only; network disabled. No
test tuning, early selection, adaptation, or candidate filtering. Record raw
strings/tokens, audio hashes and per-utterance decode time. Use saved predictions
for scoring and resume only under identical model/source/config/data identities.

The first registered TRAIN recording must complete the same runner and scorer,
with the original target serialization matching its saved output, before the
full test starts. This checks wiring only and does not establish accuracy.
The test has exactly one fixed inference pass; the compute allowance is up to
24 CPU-core-hours (four cores × six hours), evaluated after completed utterances.
If this allowance is reached, retain partial coverage and do not report a full
test reproduction. Engine/process errors retain all evidence and require
inspection, not automatic retries with changed settings.

Scoring preserves the official `prepare.py` serialization, including
`{"action:" "..."| "object": "..."| "location": "..."}`. Report exact original
serialization and joint action/object/location accuracy, with component
accuracy separately. Literal `none` is not unknown/null. Missing output is wrong
in the full split denominator; duplicate/unknown IDs fail. Do not score this
legacy serialization as Jev JSON. The current upstream recipe reports semantic
WER/CER; our native joint metric is independently implemented and is not labeled
as that recipe's WER. The publisher's exact historical accuracy computation is
not independently verified, so expose both exact/native metrics and any gap.

Full completion requires Docker/application exit 0, 3,793 unique predictions,
matching source IDs/hashes, and independent reaggregation. Report the difference
from 99.6% without declaring an arbitrary acceptance tolerance. This baseline
belongs only to the native FSC table, not schema rebinding or cross-task transfer.
