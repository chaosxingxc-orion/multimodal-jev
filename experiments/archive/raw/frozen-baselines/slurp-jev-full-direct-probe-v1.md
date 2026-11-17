# Full SLURP-Jev direct consumption probe

Freeze before inference: first group from previously locked native TRAIN16 list
(`slurp:12507`), all2 recordings, all168 decisions each (336 requests).
Use full bundle release2cf048949e551884179516487b8f8809e4f19621381146f9dd64ac1c392596eb,
resolved by label-blind materializer. Full60-way intent options remain intact.
No teacher prompt/candidate is available; this is baseline interface verification.

Qwen3-ASR-0.6B-hf, frozen official weights, eager BF16, seed17, greedy n1,
max256 output tokens, context4096, same pinned NPU image52ba94e1, physical175
card4. Use frozen-multivalue-decision-prompt-v1.txt; original assistant-prefix
JSON generation. No transcript phase, separate model, trained readout or repair.
One hour app limit/3900s outer, immutable snapshot, hashes before/after.

Require all336 requests and full2 audio bundles. Report typed format/schema,
intent views, complementary presence, empty/nonempty list exact separately;
reconstruct native intent/entity metrics through unchanged official scorer.
Malformed extraction makes native reconstructed output invalid; preserve all
records in denominator. This2-recording TRAIN engineering probe is not official
SLURP performance or evidence of cross-task generalization. No full test queue.
