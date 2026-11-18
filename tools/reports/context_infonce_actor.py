"""Reuse the pinned verl actor with a supervised InfoNCE loss hook only."""
from contextvars import ContextVar
import hashlib
from pathlib import Path
import types

from multimodal_jev.context_infonce import loss

ACTOR_SHA = '067297bc726d2b607776d40cf6e155f98649ba9405f1d025b43d803e9c1215b0'
MODE = 'jev_context_infonce_v1'


def load(source_root, *, temperature):
    path = Path(source_root) / 'src/multimodal_jev/context_native_actor.py'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ACTOR_SHA:
        raise ValueError('pinned native actor changed')
    module = types.ModuleType('multimodal_jev._infonce_actor')
    module.__package__ = 'multimodal_jev'
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    module.MODE = MODE
    module._active = ContextVar('jev_infonce_active', default=None)

    def instruction_loss(logits, labels, observed, weights, real, **kwargs):
        if (kwargs['stage'] != 'instruction' or kwargs['ranking_coefficient'] != 1.
                or kwargs['probability_coefficient'] != 0. or kwargs['kl_coefficient'] != 0.
                or any(kwargs[k] is not None for k in
                       ('actions', 'old_joint', 'advantages', 'reference_policy'))):
            raise ValueError('pure supervised InfoNCE stage without rollout/reference required')
        return loss(logits, labels, observed, weights, real,
            global_real_requests=kwargs['global_real_requests'],
            world_size=kwargs['world_size'], temperature=temperature)

    module.loss = instruction_loss
    return module
