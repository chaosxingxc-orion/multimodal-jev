"""Isolated fixed-projection/packed-attention candidate, never a production profile.

Install once in a fresh process, before FSDP wrapping. The returned handle must
remain alive. Only complete right-padded prefills with dropout zero are supported.
"""
from types import MethodType

import torch

from .span_fixed_shape_linear import fixed_shape_linear
from .span_packed_attention import packed_causal_attention, masked_row_varlen_kernel


def install_candidate(adapted, span_model, *, shared_splits=False):
    import transformers.models.qwen3.modeling_qwen3 as qwen3
    if getattr(qwen3.eager_attention_forward, '_jev_batch_candidate', False):
        raise ValueError('candidate already installed in this process')
    patched = []

    def projection(module, inputs):
        if inputs.ndim not in (2, 3):
            raise ValueError('candidate supports 2D/3D Linear only')
        values = inputs[None] if inputs.ndim == 2 else inputs
        out = fixed_shape_linear(values, module.weight, module.bias, tile_rows=128, tile_batch=8,
                                 shared_splits=shared_splits)
        return out[0] if inputs.ndim == 2 else out

    for name, module in adapted.named_modules():
        if type(module) is torch.nn.Linear:
            if module.weight.dtype != torch.bfloat16:
                raise ValueError('install before FP32 master creation/FSDP')
            module.forward = MethodType(projection, module)
            patched.append(name)
    if not any('lora_A.' in name for name in patched):
        raise ValueError('LoRA leaves required')
    active = {}

    def lengths(module, args, kwargs):
        mask = kwargs['attention_mask']
        if mask.ndim != 2 or not ((mask == 0) | (mask == 1)).all():
            raise ValueError('binary full-prefill mask required')
        if not torch.equal(mask, mask.cumprod(-1)):
            raise ValueError('right-padding only')
        active['lengths'] = [int(n) for n in mask.sum(-1).cpu().tolist()]

    def attention(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
        if dropout != 0. or 'lengths' not in active:
            raise ValueError('explicit deterministic complete prefills required')
        return packed_causal_attention(query, key, value, active['lengths'], scaling=scaling,
                                       kernel=masked_row_varlen_kernel, shared_splits=shared_splits), None

    attention._jev_batch_candidate = True
    qwen3.eager_attention_forward = attention
    handle = span_model.backbone.register_forward_pre_hook(lengths, with_kwargs=True)
    return handle, dict(patched_linear_names=patched, tile_rows=128, tile_batch=8,
                        attention='TND_sparse3_inner2', production_promoted=False)
