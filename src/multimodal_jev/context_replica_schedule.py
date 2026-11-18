"""Rebatch authenticated source epochs with 64 queries per DP replica.

Preserve real sample order exactly; remove old padding before adding new tails.
Runtime rank assignment must use actual encoded lengths, not inherited DP4 ranks.
"""
from collections import Counter

from .context_topology_batch_plan import derive
from .span_length_groups import length_group_indices


def rebatch(parent_batches, population, config):
    plan = derive(config)
    if len(population) != config['source_queries'] or len(set(population)) != len(population):
        raise ValueError('complete distinct source population required')
    expected = Counter(population)
    if {b['epoch'] for b in parent_batches} != set(range(config['epochs'])):
        raise ValueError('complete declared epochs required')
    result = []
    for epoch in range(config['epochs']):
        original = [b for b in parent_batches if b['epoch'] == epoch]
        if [b['index'] for b in original] != list(range(len(original))):
            raise ValueError('ordered contiguous source batches required')
        real = []
        for batch in original:
            for slot in batch['slots']:
                if (set(slot) != {'id','weight','padding'} or slot['id'] not in expected
                        or type(slot['padding']) is not bool
                        or type(slot['weight']) not in (int,float)
                        or slot['weight'] != float(not slot['padding'])):
                    raise ValueError('known uniform real and zero padding slots required')
                if not slot['padding']: real.append(dict(slot))
        if Counter(s['id'] for s in real) != expected:
            raise ValueError('source epoch coverage differs')
        slots = list(real)
        while len(slots) % plan['global_batch']:
            slots.append(dict(real[(len(slots)-len(real)) % len(real)],weight=0.,padding=True))
        for index,start in enumerate(range(0,len(slots),plan['global_batch'])):
            group = slots[start:start+plan['global_batch']]
            result.append(dict(epoch=epoch,index=index,slots=group,
                real_queries=sum(not s['padding'] for s in group)))
    return result


def rank_indices(batch, lengths, config):
    plan = derive(config)
    if len(batch['slots']) != plan['global_batch'] or len(lengths) != plan['global_batch']:
        raise ValueError('complete macro and actual encoded lengths required')
    return length_group_indices(lengths, microbatch=config['microbatch'],
        world_size=plan['dp_replicas'], order='length_grouped')


def load_schedule(root, manifest_sha256, *, release_sha256, population,
                  batch_config, parent_root, parent_manifest_sha256, parent_release_sha256):
    """Rebuild from an authenticated parent, including order and padding IDs."""
    import json
    from pathlib import Path
    from .baseline import file_hash
    from .context_training_schedule import load_schedule as load_parent
    from .slurp_training import content_hash
    root=Path(root)
    if file_hash(root/'MANIFEST.json') != manifest_sha256:
        raise ValueError('replica schedule manifest changed')
    manifest=json.loads((root/'MANIFEST.json').read_text())
    if (manifest['schema'] != 'jev-replica-source-schedule-v1'
            or manifest['release_sha256'] != release_sha256
            or manifest['parent_manifest_sha256'] != parent_manifest_sha256
            or manifest['parent_release_sha256'] != parent_release_sha256
            or manifest['batch_plan'] != derive(batch_config)
            or manifest['real_population_sha256'] != content_hash(sorted(population))
            or set(manifest['files']) != {'batches.json'}
            or file_hash(root/'batches.json') != manifest['files']['batches.json']):
        raise ValueError('source population, topology or schedule bytes differ')
    parent,_=load_parent(parent_root,parent_manifest_sha256,
        release_sha256=parent_release_sha256,population=population)
    expected=rebatch(parent,list(population),batch_config)
    actual=json.loads((root/'batches.json').read_text())
    if (actual != expected or manifest['total_updates'] != len(expected)
            or manifest['total_real_exposures'] != sum(b['real_queries'] for b in expected)):
        raise ValueError('schedule does not reproduce authenticated source order and weights')
    return actual,manifest
