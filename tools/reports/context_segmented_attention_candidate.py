"""Independent full-prefill component for a future batched rollout profile.

No installer or scheduler override: the caller must supply authenticated complete
request lengths from scheduler metadata. Position resets alone cannot prove that
a request is complete. Cached/chunked prefills and decoding are out of scope.
The existing production profile is deliberately untouched.
"""
import math

import torch

from multimodal_jev.span_packed_attention import masked_row_varlen_kernel


def validate_segments(positions, lengths, total):
    if (type(lengths) not in (list, tuple) or not lengths
            or any(type(n) is not int or n < 1 for n in lengths)
            or sum(lengths) != total or total >= 2**31
            or positions.ndim != 1 or len(positions) != total
            or positions.dtype not in (torch.int32, torch.int64)):
        raise ValueError('explicit complete request lengths and integer positions required')
    expected = torch.cat([torch.arange(n, device=positions.device,
                                      dtype=positions.dtype) for n in lengths])
    if not torch.equal(positions, expected):
        raise ValueError('each complete prefill must start at zero and increase by one')


def segmented_causal_attention(query, key, value, positions, lengths, *, scaling, kernel=None):
    """Operate directly on flat T,H,D; no B*max_length padded allocation.

    The injected CPU kernel is only a component oracle. Default execution uses
    the existing BF16 Ascend variable-length kernel and does not grant admission.
    """
    if (query.ndim != 3 or key.ndim != 3 or value.shape != key.shape
            or min(query.shape) < 1 or min(key.shape) < 1
            or query.shape[0] != key.shape[0] or query.shape[2] != key.shape[2]
            or query.shape[1] % key.shape[1] or not query.is_floating_point()
            or any(t.device != query.device or t.dtype != query.dtype for t in (key, value))
            or positions.device != query.device
            or type(scaling) not in (int, float) or not math.isfinite(scaling) or scaling <= 0):
        raise ValueError('aligned flat full-prefill Q/K/V required')
    validate_segments(positions, lengths, len(query))
    if kernel is None:
        if query.device.type != 'npu' or query.dtype != torch.bfloat16:
            raise ValueError('default segmented kernel requires BF16 Ascend tensors')
        kernel = masked_row_varlen_kernel
    cumulative = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0, dtype=torch.int32)
    result = kernel(query, key, value, cumulative, cumulative,
                    max_seqlen_q=max(lengths), max_seqlen_k=max(lengths),
                    dropout_p=0., softmax_scale=scaling, causal=True)
    if result.shape != query.shape or result.dtype != query.dtype or result.device != query.device:
        raise ValueError('segmented attention returned incompatible output')
    return result


def segmented_attention_forward(module, positions, hidden_states, lengths, *, kernel=None):
    """Candidate vLLM projection flow with an explicit per-call length contract.

    Keeping lengths explicit avoids global mutable state leaking across requests.
    A future worker adapter must bind them to the actual scheduler batch, and
    separately enforce TP1/no-prefix-cache/no-chunking and policy identity.
    """
    if hidden_states.ndim != 2 or not hidden_states.is_floating_point():
        raise ValueError('flat floating-point hidden states required')
    validate_segments(positions, lengths, len(hidden_states))
    if positions.device != hidden_states.device:
        raise ValueError('positions and hidden states must share a device')
    qkv, _ = module.qkv_proj(hidden_states)
    q, k, v = qkv.split([module.q_size, module.kv_size, module.kv_size], dim=-1)
    q = module.q_norm(q.reshape(len(q), module.num_heads, module.head_dim)).reshape(q.shape)
    k = module.k_norm(k.reshape(len(k), module.num_kv_heads, module.head_dim)).reshape(k.shape)
    q, k = module.rotary_emb(positions, q, k)
    q, k, v = [x.reshape(len(x), heads, module.head_dim) for x, heads in
               ((q, module.num_heads), (k, module.num_kv_heads), (v, module.num_kv_heads))]
    attended = segmented_causal_attention(q, k, v, positions, lengths,
                                         scaling=module.scaling, kernel=kernel)
    return module.o_proj(attended.reshape(len(hidden_states), module.q_size))[0]
