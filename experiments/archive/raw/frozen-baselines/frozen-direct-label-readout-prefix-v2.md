# Frozen ASR output-position diagnostic v2

The v1 TRAIN64 six-view readout chose position 28 for all 256 choice requests
under both actual and zero audio. No discriminating task success was established.
The pinned Qwen3-ASR-0.6B-hf README (Chat template section) specifies a fixed
assistant prefill `language English<asr_text>` with `continue_final_message=True`.
V1 read the first assistant position, before that protocol header. Test whether
the fixed header changes task readout behavior without altering any weights.

Use exactly the original 64 TRAIN groups and 384 six-view requests, same v1 task
instruction, model, BF16 eager runtime, seed17, context4096 and candidate mapping.
Only add the fixed assistant header. It contains no transcript, label, gold or
audio-dependent text. No ASR generation stage or second model is invoked.
This is a new decoder interface diagnostic, not teacher-produced prompt improvement.

Run actual waveform and equal-duration zero waveform on physical card4 in order,
3600 app seconds / 3900 outer seconds each, immutable source snapshots and pinned
image52ba94e1. Preserve hashes, probabilities, logits, typed values and complete
six-view scoring. Compare paired values and position histograms as v1. Zero audio
is only a sensitivity control; format enforcement is not learned format ability.
Do not tune from selection/test data. Do not change any prior parity threshold.
No training or teacher dispatch in this diagnostic. If discrimination still fails,
retain the failure and evaluate the suitability of the checkpoint for frozen
task prompting; do not silently substitute a transcript cascade or trained head.
