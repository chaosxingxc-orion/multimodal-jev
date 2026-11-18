"""Opt-in vLLM component probe for the actor's packed attention kernel.

TP1, one complete uncached prefill only. This is not a production profile.
Uses live vLLM weights, native LoRA and the existing aligned projections.
"""
from types import MethodType

import torch

from .lora_span_audit import SpanAuditWorkerExtension
from .span_packed_attention import packed_causal_attention, masked_row_varlen_kernel


def packed_attention_forward(module, positions, hidden_states):
    if (positions.ndim != 1 or hidden_states.ndim != 2
            or len(positions) != len(hidden_states) or not len(positions)
            or not torch.equal(positions, torch.arange(len(positions), device=positions.device))):
        raise ValueError('single complete uncached prefill starting at zero required')
    qkv, _ = module.qkv_proj(hidden_states)
    q, k, v = qkv.split([module.q_size, module.kv_size, module.kv_size], dim=-1)
    q = module.q_norm(q.view(len(q), module.num_heads, module.head_dim)).view(q.shape)
    k = module.k_norm(k.view(len(k), module.num_kv_heads, module.head_dim)).view(k.shape)
    q, k = module.rotary_emb(positions, q, k)
    tensors = [x.view(len(x), heads, module.head_dim).transpose(0, 1).unsqueeze(0)
               for x, heads in ((q, module.num_heads), (k, module.num_kv_heads),
                                (v, module.num_kv_heads))]
    attended = packed_causal_attention(*tensors, [len(positions)], scaling=module.scaling,
                                      kernel=masked_row_varlen_kernel)
    output, _ = module.o_proj(attended.reshape(len(hidden_states), module.q_size))
    return output


def install_packed_rollout_attention(model):
    layers = model.language_model.model.layers
    if not layers:
        raise ValueError('nonempty language model required')
    for layer in layers:
        module = layer.self_attn
        if module.qkv_proj.tp_size != 1 or getattr(module, '_jev_packed_candidate', False):
            raise ValueError('fresh TP1 attention required')
    for layer in layers:
        layer.self_attn.forward = MethodType(packed_attention_forward, layer.self_attn)
        layer.self_attn._jev_packed_candidate = True
    return dict(layers=len(layers), attention='TND_sparse3_inner2',
                production_promoted=False, scope='single complete prefill; attention component only')


class BatchCandidateWorkerExtension(SpanAuditWorkerExtension):
    def align_jev_candidate_packed_attention(self):
        return install_packed_rollout_attention(self.get_model())

    def align_jev_candidate_fixed_base(self):
        from .span_fixed_rollout_projection import install_fixed_rollout_projections
        return install_fixed_rollout_projections(self.get_model())
