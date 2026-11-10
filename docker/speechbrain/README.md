# SpeechBrain baseline environment

An additive image over the pinned NGA-derived runtime, with a separate virtual
environment exposing the base torch/torchaudio libraries. Source is pinned at
`c0241278f93ef02cd5191f03caf4e5f8e1292bf6` (Apache-2.0). This is a current official
inference port; it does not reproduce the historical FSC training environment.

Build context and hashes are preserved under shared experiments at
`multimodal-jev/speechbrain-runtime-build-v1`. Only the four extra wheels in
`requirements-extra.txt` are installed, with source SHA256 verification and no
dependency resolution/network. Keep this environment separate from RL images.
The build does not establish import, model loading or task accuracy; those gates
must be checked and reported separately. Inherited packages still require a
targeted dependency audit, since a successful no-deps installation is insufficient.

FSC inference must pin both the SLU model and its recursively loaded LibriSpeech
ASR dependency, overriding `asr_model_source` to the verified local path.
Keep the original beam size 80 and temperature 1.25; preserve raw predictions
and original native action/object/location scoring. The published 99.6% test
accuracy is an author report, not a local result. See the
[asset receipt](../../docs/engineering/fsc-speechbrain-assets-v1.json).
