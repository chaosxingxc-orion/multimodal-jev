#!/usr/bin/env python3
"""Real tiny Qwen3-ASR eager CPU batching check; no pretrained/NPU claim."""
import copy
import json

import torch
from transformers import Qwen3ASRConfig, Qwen3ASRForConditionalGeneration

from multimodal_jev.span_audio_collate import forward_span_audio_batch
from multimodal_jev.span_model import NativeAudioSpanModel


def main():
    torch.set_num_threads(2)
    torch.manual_seed(317)
    config = Qwen3ASRConfig(
        audio_config=dict(num_mel_bins=8, encoder_layers=1, encoder_attention_heads=2,
                          encoder_ffn_dim=32, d_model=16, output_dim=16,
                          downsample_hidden_size=4, n_window=50, n_window_infer=800),
        text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=8, max_position_embeddings=1024),
        audio_token_id=9, timestamp_token_id=10, pad_token_id=0, eos_token_id=3)
    base = Qwen3ASRForConditionalGeneration(config)
    base.set_attn_implementation('eager')
    model = NativeAudioSpanModel(base).eval()
    control = copy.deepcopy(model)
    rows = []
    for frames in (31, 100, 205, 899):
        full, tail = divmod(frames, 100)
        count = full * 13 + (tail + 7) // 8
        ids = torch.tensor([[1] + [9] * count + [2, 4, 5]])
        width = ((frames + 99) // 100) * 100
        rows.append(dict(inputs=dict(input_ids=ids, attention_mask=torch.ones_like(ids),
            input_features=torch.randn(1, 8, width),
            input_features_mask=(torch.arange(width)[None] < frames).long()),
            positions=torch.tensor([[ids.shape[1] - 1]]), candidates=[[2, 4, 7]]))
    calls = []
    hook = model.backbone.register_forward_hook(lambda *args: calls.append(1))
    actual = forward_span_audio_batch(model, rows, device='cpu', dtype=torch.float32,
        pad_token_id=0, audio_token_id=9, audio_chunk_size=100, audio_infer_window=800)
    hook.remove()
    expected = [control(r['inputs'], r['positions'], r['candidates'])[1] for r in rows]
    assert len(calls) == 1
    errors = []
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a[0], b[0], atol=2e-6, rtol=2e-6)
        errors.append(float((a[0] - b[0]).abs().max().detach()))
    # Include an explicit zero-weight padding request and nonuniform class weights.
    weights = [1., 2., .5, 0.]
    def loss(values):
        return sum(-w * p[0][0, i % 3] / 3 for i, (w, p) in enumerate(zip(weights, values)))
    la, lb = loss(actual), loss(expected)
    torch.testing.assert_close(la, lb, atol=2e-6, rtol=2e-6)
    la.backward(); lb.backward()
    grad_error = 0.
    for a, b in zip(model.parameters(), control.parameters()):
        if a.grad is None or b.grad is None:
            assert a.grad is None and b.grad is None
            continue
        torch.testing.assert_close(a.grad, b.grad, atol=5e-6, rtol=5e-5)
        grad_error = max(grad_error, float((a.grad - b.grad).abs().max()))
    print(json.dumps(dict(passed=True, backbone_calls=len(calls), requests=len(rows),
        max_logprob_error=max(errors), max_gradient_error=grad_error,
        scope='Real tiny randomly initialized Qwen3-ASR eager FP32 CPU; synthetic features; not pretrained BF16 NPU, LoRA/FSDP or vLLM admission.')))


if __name__ == '__main__':
    main()
