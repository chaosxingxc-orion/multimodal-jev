# Direct audio frozen prompt optimization: failure reflection v1

2026-11-10. User priority is frozen base inference, remote-teacher prompt
optimization, then analogous xx-jev optimization; no weight training.
This new protocol uses the completed direct-audio fixed-ASR-header diagnostic
as parent. It is ACE-inspired failure reflection, not a claim of TF-GRPO
semantic advantage: no successful paired Noul discrimination was observed.

Actor: official Qwen3-ASR-0.6B-hf, frozen, HF eager BF16, seed17, physical175
cards4–7 only. Input is audio plus question/alternatives; fixed assistant
`language English<asr_text>`; candidate next-token readout with unchanged
0–9,A–Z label mapping. No transcription stage, second online model or head.
Parent and candidate share this exact interface, including optional null.
The native FSC view is the original 31-way intent task; all six views are a
separate correlated FSC-Jev task table. Open SLURP is outside this decoder.

Teacher learns only from the previously locked64 TRAIN recording groups,
all384 requests, completed actual and zero runs, strict scores and bounded
representative predictions. Gold is offline feedback. Do not send audio,
reference transcripts, private reasoning, credentials, or held-out feedback.
Teacher has not listened to audio. Silence is a sensitivity control, not a
gold unknown task. No invented mixed-success group sampling or advantages.

One durable glm-5.3-flash request, enabled thinking, max32768 output tokens,
900 seconds total, at most4 replacement instructions each <=4000 characters.
Reuse Omni executor SHA c9f9bbd333f038955945db82d946cb2757bafbc340b1a8ca476f52f3ab406bd4.
No automatic retry, repair call, source execution or candidate promotion.
Keep actual usage and unknown/failed results; old cascade costs stay separate.
Local review rejects memorized example/candidate mappings and modified label,
model, candidate, null, scorer or interface semantics. Teacher modifies only
the parent task instruction; task definitions remain supplied dynamically.

After local validation/review, evaluate parent and all accepted candidates on
the fixed next256 disjoint TRAIN groups (1536 requests each), actual audio,
same deterministic readout. Choose lexicographically by all-six-correct,
then mean of four paired both-correct rates, then mean of six per-view
accuracies; exact ties keep parent, novel ties prefer shorter then ID.
Keep all regressions and per-view scores. No claim of improvement from format
alone. At most5 selection runs, each3600 app seconds/3900 outer seconds.
Selected prompt then freezes for full official TEST, paired to the unchanged
parent; no TEST feedback enters any prompt or selection. Cross-dataset
generalization requires additional source-held-out datasets, not these views.
