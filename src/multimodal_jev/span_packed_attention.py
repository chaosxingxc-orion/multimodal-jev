"""Experimental full-prefill variable-length causal attention on Ascend.

Explicit lengths describe right-padded independent requests. No decoding,
prefix sharing, custom masks or runtime admission is provided by this helper.
"""
import math

import torch
from torch.nn import functional as F

from . import card_attention_groups as card_groups


def masked_row_varlen_kernel(q, k, v, cq, ck, **kwargs):
    """Candidate Ascend variant; inner_precise=2 handles invalid masked rows.

    This is not a request for FP32 accumulation or a production profile.
    """
    import torch_npu
    mask = card_groups.causal_mask(q.device) if card_groups.active() else torch.ones((2048, 2048), device=q.device, dtype=torch.bool).triu(1)
    return torch_npu.npu_fusion_attention(
        q, k, v, q.shape[1], "TND", atten_mask=mask,
        scale=kwargs['softmax_scale'], keep_prob=1.,
        actual_seq_qlen=tuple(cq[1:].tolist()),
        actual_seq_kvlen=tuple(ck[1:].tolist()),
        sparse_mode=3, inner_precise=2, next_tockens=0)[0]


def packed_causal_attention(query, key, value, lengths, *, scaling, kernel=None, shared_splits=False):
    """Pack B,H,S,D into T,H,D, invoke one varlen kernel, restore B,S,H,D.

    A supplied kernel is for independent CPU contract tests. The default uses
    the pinned Transformers Ascend wrapper, including its causal mask handling.
    """
    if (query.ndim != 4 or key.ndim != 4 or value.shape != key.shape
            or query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]
            or min(query.shape) < 1 or min(key.shape) < 1
            or query.shape[1] % key.shape[1]
            or any(t.dtype != query.dtype or t.device != query.device for t in (key, value))
            or not query.is_floating_point()
            or not math.isfinite(scaling) or scaling <= 0
            or len(lengths) != query.shape[0]
            or any(type(n) is not int or not 0 < n <= query.shape[2] for n in lengths)):
        raise ValueError('aligned full-prefill Q/K/V and positive right-padding lengths required')
    if kernel is None:
        if query.device.type != 'npu' or query.dtype != torch.bfloat16:
            raise ValueError('default experimental kernel requires BF16 Ascend tensors')
        from transformers.integrations.npu_flash_attention import npu_flash_attn_varlen_func
        kernel = npu_flash_attn_varlen_func
    if card_groups.active():
        return card_groups.grouped(query, key, value, lengths, scaling, kernel)
    def pack(t):
        if shared_splits:
            return torch.cat([row[:, :n].transpose(0, 1) for row, n in
                              zip(t.unbind(0), lengths, strict=True)])
        return torch.cat([t[i, :, :n].transpose(0, 1) for i, n in enumerate(lengths)])
    packed = [pack(t) for t in (query, key, value)]
    cumulative = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0, dtype=torch.int32)
    output = kernel(*packed, cumulative, cumulative, max_seqlen_q=max(lengths),
                    max_seqlen_k=max(lengths), dropout_p=0., softmax_scale=scaling, causal=True)
    if output.shape != packed[0].shape:
        raise ValueError('packed attention returned unexpected shape')
    return torch.stack([F.pad(part, (0, 0, 0, 0, 0, query.shape[2]-n))
                        for part, n in zip(output.split(lengths), lengths, strict=True)], 0)
