# NPU sampler lifecycle correction v2

v1 completed 6 prompts / 12 responses / 80 tokens and wrote application success,
but did not exit its Docker process. Read-only py-spy confirmed interpreter exit
waiting in multiprocessing `BaseProcess.join`; the vLLM engine remained live.
This is distinct from a generation timeout. Preserve v1 evidence and process
until an explicit cleanup decision, not a restart from stale progress.

v2 repeats the byte-identical BF16 config and checkpoint on allocated physical
NPU 5, with the same pinned image. The only sampler change is an explicit
`engine.llm_engine.engine_core.shutdown(timeout=30)` after persisted responses
and before result/application success. Test actual Docker exit, not just the
application file. Frozen CPU experiments keep their old snapshots.

v1's separate physical-NPU-5 HF BF16 audit failed at max log-probability error
0.1861983 versus the unchanged 0.02 gate, all 12 responses finite. v2 does not
resolve or hide this mismatch. Do not launch RL updates based on this sampler
until a separately evidenced numerical fix passes the original probability gate.
