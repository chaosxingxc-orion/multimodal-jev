"""Prepare the next scheduled update from accepted behavior and fixed reference.

The caller authenticates execution/checkpoint receipts. This interface does not
certify a backend or advance a training cursor; it binds their actual records
to the next full learning batch before the native DP actor receives targets.
"""
from .simple_audio_batch import compile_batch, renderer_for_policy
from .simple_sampled_rollout import actor_batch


def next_batch(population, task, slots, processor, *, behavior_result,
               reference_result, policy_id, reference_policy_id, completed_updates,
               seed, global_batch=128):
    if (type(completed_updates) is not int or completed_updates < 1
            or policy_id == reference_policy_id):
        raise ValueError('updated behavior and positive completed RL cursor required')
    entries, renderer, compiled = compile_batch(population, task, slots, processor,
        policy_id=policy_id, global_batch=global_batch)
    reference_renderer = renderer_for_policy(renderer, reference_policy_id)

    def select(result, rows, expected_policy):
        if (result.get('passed') is not True or result.get('backend') != 'vllm'
                or result.get('base_id', '')+':'+result.get('adapter_id', '') != expected_policy):
            raise ValueError('successful current behavior or fixed reference required')
        records = result['records']
        by_id = {row['request_id']: row for row in records}
        if len(by_id) != len(records):
            raise ValueError('duplicate backend request in accepted population')
        requested = [row[4] for row in rows]
        if any(rid not in by_id for rid in requested):
            raise ValueError('next scheduled batch missing from accepted backend')
        return dict(result, records=[by_id[rid] for rid in requested])

    behavior = select(behavior_result, renderer, policy_id)
    reference = select(reference_result, reference_renderer, reference_policy_id)
    items, sampling = actor_batch(entries, behavior, reference, policy_id=policy_id,
        reference_policy_id=reference_policy_id, seed=seed, update=completed_updates)
    return items, dict(compiled=compiled, sampling=sampling,
        next_optimizer_update=completed_updates+1,
        scope='Next full scheduled learning batch using updated behavior and unchanged reference; no optimizer execution or admission.')
