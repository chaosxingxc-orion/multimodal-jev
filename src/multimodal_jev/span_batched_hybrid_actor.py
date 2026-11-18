"""Opt-in experimental true-batch forward for the unchanged verl hybrid loop.

The existing builder still authenticates verl, validates every update and owns
the shared loss context. This replaces only the forward on a fresh actor, never
an already running trainer. Production entrypoints do not select this builder.
"""
from types import MethodType

import torch

from .span_audio_collate import forward_span_audio_batch
from .span_parity import compare_span_policies


def batched_forward(actor, micro_batch, temperature, calculate_entropy=False):
    from .span_hybrid_actor import _active
    if temperature != 1. or calculate_entropy:
        raise ValueError('untransformed typed forward required')
    state = _active.get()
    if state is not None and state.get('log_probs') is not None and not state['consumed']:
        raise ValueError('previous hybrid forward not consumed')
    payload = list(micro_batch['multi_modal_inputs'])
    if not payload or any(p['policy_id'] != actor.config.jev_policy_id for p in payload):
        raise ValueError('nonempty current-policy batch required')
    # Forward helper reads only inputs, positions and candidate support, not gold.
    outputs = forward_span_audio_batch(actor.actor_module, payload,
        device=next(actor.actor_module.parameters()).device,
        dtype=actor.param_dtype, **actor.jev_audio_collation)
    packed = []
    for p, logp in zip(payload, outputs, strict=True):
        if p['stage'] == 'task' and not compare_span_policies(
                [v[0].detach().cpu().tolist() for v in logp],
                [v[0].tolist() for v in p['behavior']])['passed']:
            raise ValueError('complete behavior parity failed')
        joined = torch.cat(logp, -1)
        if p['width'] < joined.shape[-1]:
            raise ValueError('candidate transport width is too small')
        packed.append(torch.nn.functional.pad(joined, (0, p['width']-joined.shape[-1])))
    joined = torch.cat(packed, 0)
    if state is not None:
        state.update(log_probs=joined, payload=payload, consumed=False)
    return {'log_probs': joined}


def build_batched_actor(config, module, optimizer, *, collation):
    from .span_hybrid_actor import build_actor
    required = {'pad_token_id', 'audio_token_id', 'audio_chunk_size', 'audio_infer_window'}
    if set(collation) != required:
        raise ValueError('explicit complete audio collation contract required')
    actor = build_actor(config, module, optimizer)
    actor.jev_audio_collation = dict(collation)
    actor._forward_micro_batch = MethodType(batched_forward, actor)
    return actor
