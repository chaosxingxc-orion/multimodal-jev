# Native SLURP frozen end-to-end probe v2

Use verified native release240070cb69d71f5df53874ff9e728611dd1bd56a564f7f9e7718092125612add,
real audio only, unique filename counts50627/8690/13078. Preserve all entity
occurrences and official scenario/action taxonomy. This replaces only the
cancelled cascade execution; no transcript/oracle input is permitted.

Select first16 source groups by ascending SHA256(group), including all recordings
per group. Reserve first64 learning and next256 selection groups from that same
order. Linked duplicate utterance groups remain atomic. Freeze groups before
dispatch. Probe has no teacher, weight training or test optimization.

Official Qwen3-ASR-0.6B-hf frozen, same pinned image52ba94e1, HF eager BF16,
seed17, greedy n1, context4096, max256 output tokens, physical175 card5.
Audio and complete task taxonomy go directly into the same model with unchanged
frozen-slurp-native-prompt-v1.txt. Use original free JSON generation interface
for consistency with native FSC frozen generation, no ASR-header stripping,
finite-label substitute, text model, output repair or hidden transcript stage.
The separate FSC prefix-readout diagnostic does not cover open SLURP entities.

One-hour app/3900 outer limit, immutable snapshot, model hashes before/after,
complete prediction and terminal records. Score with pinned official SLURP
metrics and strict complete-ID adapter. Malformed output counts as invalid
scenario/action and empty entities; never exclude malformed records.
Report TRAIN probe coverage, format rate, intent and entity metrics; no full
official TEST score or prompt improvement claim. Failure remains valid baseline
evidence. Follow with a separately frozen full-data condition after interface
verification; no automatic full run is authorized by this probe.
