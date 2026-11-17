# Cached HF response diagnosis v1

Use the 12 saved BF16 NPU responses from npu-vllm-audio-v1 and the identical
merged initializer in HF BF16 on allocated physical NPU 7. No new sampling,
training, label-based selection or threshold change. Prefill the exact prompt
and audio once, then teacher-force each saved response token with past_key_values,
causal position IDs and growing attention masks. Include every response token.

Compare with original vLLM sampled log probabilities at the existing 0.02 gate,
and with saved full-sequence HF BF16 results. This isolates cache/forward-shape
sensitivity more directly than changing model precision alone. Passing only
cached inference does not admit full-sequence training or prove a kernel bug.
Keep original full-sequence failures and all per-token probabilities.

The causal boundary test uses explicit token-transition logits and verifies
prefill-only audio, reused cache, growing masks and response positions. Actual
Qwen execution remains necessary beyond that controlled test.
