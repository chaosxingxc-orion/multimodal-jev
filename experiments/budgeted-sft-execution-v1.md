# Real SFT budget-stop validation

Use the first two complete six-view batches from seed17's frozen comparison
schedule, shared SFT initializer, original base and the existing CPU float32
SFT runner. This is an execution check, not a quality comparison. Do not select
recordings by length, loss or development correctness.

Pin the successful train-pool-budget-profile-v2 result by SHA256 after its
exact Docker exit0. Set the training-token limit to the measured exact gold
input-token total of the first batch, generated-token limit zero. Expect one
complete optimizer update, then budget_exhausted before the second batch,
with zero remaining training token units and six executed examples. Check
finite loss and gradient norm, saved adapter change from the initializer,
resume history length/step1 and finite optimizer step1. Verify output counts
against the source profile and the saved training budget independently.

Run in the pinned NGA image on remote CPU only, four CPU threads, 32 GiB
memory, no NPU devices. Retain immutable source/config, exact Docker ID,
application terminal and failure evidence. A timeout is observation failure,
not permission to restart. No automatic retry and no promotion to the formal
comparison without successful result and saved-state checks. This does not
validate the multi-step RL budget path, which remains a separate required gate.

## Completed execution

Both exact training and audit containers exited0. The
[result](../docs/engineering/budgeted-sft-execution-v1-result.json) records one
update over all six views: 4,127 prompt tokens plus 42 gold response tokens,
1,890 audio frames, loss0.1440768683 and pre-clipping gradient norm6.713084221.
The cap of4,169 training tokens was consumed exactly; the second batch's4,067
token reservation was rejected before its model work. Result counters report
one of two scheduled updates and `budget_exhausted`.

The separate read-only audit loaded the saved adapter and resume state:
112 adapter tensors changed from the initializer;112 finite AdamW states all
have step1. Saved history/step and result hashes agree. Host-side arithmetic
independently reproduced both batch costs from the pinned processor profile.
This verifies the real SFT stopping path, not an efficacy comparison or the
pending RL budget gate.
