"""Bind supervised InfoNCE to the shared native actor."""
from functools import partial
from pathlib import Path
from types import SimpleNamespace

from multimodal_jev import context_native_actor as native
from multimodal_jev.context_infonce import loss, validate_margin

MODE = 'jev_context_infonce_v1'
MARGIN_METRICS = (
    'infonce_numerator',
    'probability_margin_numerator',
    'threshold_anchor_numerator',
    'probability_gap_sum',
    'margin_satisfied_count',
    'anchored_count',
    'local_real_queries',
)


def load(source_root, *, temperature, probability_margin=None):
    expected_actor = Path(source_root) / 'src' / 'multimodal_jev' / 'context_native_actor.py'
    if not expected_actor.is_file() or not expected_actor.samefile(native.__file__):
        raise ValueError('native actor must be imported from the declared source root')
    if probability_margin is not None:
        validate_margin(probability_margin)

    def instruction_loss(logits, labels, observed, weights, real, **kwargs):
        if (kwargs['stage'] != 'instruction' or kwargs['ranking_coefficient'] != 1.
                or kwargs['probability_coefficient'] != 0. or kwargs['kl_coefficient'] != 0.
                or any(kwargs[k] is not None for k in
                       ('actions', 'old_joint', 'advantages', 'reference_policy'))):
            raise ValueError('pure supervised InfoNCE stage without rollout/reference required')
        return loss(
            logits, labels, observed, weights, real,
            global_real_requests=kwargs['global_real_requests'],
            world_size=kwargs['world_size'], temperature=temperature,
            probability_margin=probability_margin,
        )

    registered_loss = partial(
        native.registered_loss, mode=MODE, loss_fn=instruction_loss,
        metric_keys=MARGIN_METRICS if probability_margin is not None else (),
    )

    def build_actor(config, model, optimizer, *, collation):
        return native.build_actor(
            config, model, optimizer, collation=collation,
            objective=(MODE, registered_loss, native.admit),
            forward=adapter.forward_context_audio_batch,
        )

    adapter = SimpleNamespace(
        MODE=MODE, loss=instruction_loss, make_batch=native.make_batch, build_actor=build_actor,
        forward_context_audio_batch=native.forward_context_audio_batch,
    )
    return adapter
