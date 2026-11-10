# Frozen Qwen3-Omni 64 learning groups: real audio and silence

Only dispatch after single-group v3 completes both application/container exit0
and six-view scoring. Keep all prior failure costs. Same local official
Qwen3-Omni-Instruct, image52ba94e, TP2/BF16/eager/seed17, dynamic task schema
and frozen-decision-prompt-v1, context4096,max128tokens,free JSON,no repair.

Use the exact original64TRAIN learning groups, in the locked partition order;
selection256 and official TEST remain untouched. Each condition384requests,
actual and equal-length zero waveform. Identical new runner snapshot. All
six views of each recording remain paired. Actor has no targets/transcripts.
Process actual and zeros sequentially on physical4/5 with independent engines;
explicit spawn and shutdown. Each3600s outer,3900s observation; no auto retry.
Do not reuse v3 as a substitute for first-group outputs from the new snapshot.

Strict scores only on real audio: six per-view, four paired both-correct,
all-six, invalid/truncated counts. Silence measures waveform sensitivity only,
not unknown detection accuracy; keep original metadata for pairing but never
pretend native labels apply to silence. Compare exact model/prompt/runner/
requests plus paired rendered input token counts and IDs. Different malformed
outputs are not counted as changed decisions. No pooled independent-sample
claim across the six views. No teacher invocation or weight training scheduled.
No performance-based early stop. Require384/384coverage and successful exact
container/application terminals per condition before final comparisons.
