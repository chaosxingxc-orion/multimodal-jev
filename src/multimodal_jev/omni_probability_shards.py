"""Explicit equal-rank probability capture and complete offline reassembly.

No collectives, sampling or padding with duplicate responses. The runner must
authenticate rank/world size and ensure all ranks execute the same plan.
"""
from .omni_probability_audit import _key, _probabilities
from .omni_probability_capture import _aligned, recompute
from .sfc_reward_batch import digest


def shard_plan(records, *, world_size, micro_batch_size):
    if (type(world_size) is not int or world_size < 2
            or type(micro_batch_size) is not int or micro_batch_size < 1
            or type(records) is not list or not records):
        raise ValueError('nonempty records and explicit multi-rank schedule required')
    keys = [_key(r) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError('unique original response roster required')
    for r in records:
        _probabilities(r['token_ids'], r['log_probs'])
    # FSDP ranks must participate in the same number of forwards. Require full
    # equal microbatches, never silently drop or repeat a response for divisibility.
    if len(records) % (world_size * micro_batch_size):
        raise ValueError('complete roster must divide into equal full rank microbatches')
    count = len(records) // world_size
    return dict(schema='omni-probability-shards-v1', records_sha256=digest(records),
                world_size=world_size, micro_batch_size=micro_batch_size,
                forwards_per_rank=count // micro_batch_size,
                rank_indices=[list(range(rank * count, (rank + 1) * count))
                              for rank in range(world_size)])


def _validate_plan(records, plan, plan_sha256):
    expected = shard_plan(records, world_size=plan['world_size'],
                          micro_batch_size=plan['micro_batch_size'])
    if digest(plan) != digest(expected) or digest(plan) != plan_sha256:
        raise ValueError('unchanged registered shard plan required')


def _local_batch(data, records, *, rank, plan, plan_sha256):
    _validate_plan(records, plan, plan_sha256)
    if type(rank) is not int or not 0 <= rank < plan['world_size']:
        raise ValueError('registered rank required')
    rows = _aligned(data, records)
    positions = {_key(record): i for i, record in enumerate(rows)}
    local_records = [records[i] for i in plan['rank_indices'][rank]]
    local = data[[positions[_key(record)] for record in local_records]]
    local.meta_info = dict(local.meta_info, records_sha256=digest(local_records))
    return local, local_records


def recompute_shard(actor, data, records, *, name, rank, plan, plan_sha256, pad_token_id):
    """Validate the full input on each rank, then execute its exact local roster.

    Retains globally padded shapes and audio features. A full DataProto reorder
    is allowed; assignment follows original record identity, not current row.
    The returned envelope is partial evidence, never a complete numerical run.
    """
    local, local_records = _local_batch(data, records, rank=rank, plan=plan,
                                       plan_sha256=plan_sha256)
    run = recompute(actor, local, local_records, name=name,
                    micro_batch_size=plan['micro_batch_size'], pad_token_id=pad_token_id)
    return dict(schema='omni-probability-shard-output-v1', rank=rank,
                plan_sha256=plan_sha256, records_sha256=digest(records), run=run)


def recompute_observed_shard(actor, data, records, *, name, rank, plan,
                             plan_sha256, pad_token_id):
    """One native observed forward on the exact rank assignment, with no retry.

    Each probability envelope must be merged with ``merge_shards`` over all
    ranks, using its own run name. Observation evidence is retained separately;
    merging probabilities does not independently authenticate raw observations
    or prove NPU numeric acceptance. Execute each rank in an isolated process.
    """
    from .omni_probability_capture import recompute_native_observed

    local, local_records = _local_batch(data, records, rank=rank, plan=plan,
                                       plan_sha256=plan_sha256)
    observation = recompute_native_observed(actor, local, local_records, name=name,
        micro_batch_size=plan['micro_batch_size'], pad_token_id=pad_token_id)
    def envelope(run):
        return dict(schema='omni-probability-shard-output-v1', rank=rank,
                    plan_sha256=plan_sha256, records_sha256=digest(records), run=run)
    return dict(native=envelope(observation['native']),
                reference=envelope(observation['reference']), observation=observation)


def merge_shards(records, shards, *, name, plan, plan_sha256):
    """Require every exact rank and response before exposing a full audit run."""
    _validate_plan(records, plan, plan_sha256)
    if (type(name) is not str or not name or type(shards) is not list
            or len(shards) != plan['world_size']):
        raise ValueError('complete rank roster and registered run name required')
    ranks = set(); responses = {}
    for shard in shards:
        rank = shard['rank']
        if (type(rank) is not int or not 0 <= rank < plan['world_size'] or rank in ranks
                or shard['schema'] != 'omni-probability-shard-output-v1'
                or shard['plan_sha256'] != plan_sha256
                or shard['records_sha256'] != digest(records) or shard['run']['name'] != name):
            raise ValueError('unique authenticated rank output required')
        ranks.add(rank)
        expected = {_key(records[i]): records[i] for i in plan['rank_indices'][rank]}
        seen = set()
        for response in shard['run']['responses']:
            key = _key(response)
            if key not in expected or key in seen:
                raise ValueError('exact assigned rank response roster required')
            seen.add(key)
            original = expected[key]
            if (response['record_sha256'] != digest(original)
                    or response['token_ids'] != original['token_ids']):
                raise ValueError('original sampled response identity changed')
            _probabilities(response['token_ids'], response['log_probs'])
            responses[key] = response
        if seen != set(expected):
            raise ValueError('rank response coverage incomplete')
    return dict(name=name, responses=[responses[_key(record)] for record in records])
