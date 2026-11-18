"""Full-population uniform-query schedule for context decisions.

Class labels only spread optimizer membership over time; they never select
input questions. Length grouping uses actual processor lengths within a batch.
"""
from collections import Counter
import hashlib

from .span_class_spread import class_spread_indices
from .span_length_groups import length_group_indices
from .slurp_training import content_hash


def load_schedule(root,manifest_sha256,*,release_sha256,population):
    import json
    from pathlib import Path
    from .baseline import file_hash
    root=Path(root)
    if file_hash(root/'MANIFEST.json')!=manifest_sha256:raise ValueError('schedule identity differs')
    manifest=json.loads((root/'MANIFEST.json').read_text())
    if (manifest['schema']!='context-full-learning-schedule-v1' or manifest['release_sha256']!=release_sha256 or
            manifest['global_batch']!=128 or manifest['world_size']!=4 or manifest['microbatch']!=8 or
            manifest['accumulation']!=4 or len(manifest['epochs'])!=2):
        raise ValueError('fixed full context schedule required')
    all_batches=[]
    for epoch,report in enumerate(manifest['epochs']):
        path=root/f'epoch-{epoch}.json'
        if file_hash(path)!=report['file_sha256']:raise ValueError('epoch bytes changed')
        data=json.loads(path.read_text());batches=data['batches'];seen=Counter()
        if len(batches)!=report['updates'] or content_hash(batches)!=report['batches_sha256']:
            raise ValueError('epoch batch identity differs')
        for i,b in enumerate(batches):
            if (b['epoch']!=epoch or b['index']!=i or len(b['slots'])!=128 or len(b['rank_indices'])!=4 or
                    any(len(v)!=32 for v in b['rank_indices']) or
                    sorted(v for r in b['rank_indices'] for v in r)!=list(range(128))):
                raise ValueError('complete DP4 batch and permutation required')
            for slot in b['slots']:
                if (slot['id'] not in population or type(slot['padding']) is not bool or
                        type(slot['weight']) not in (int,float) or slot['weight']!=(0. if slot['padding'] else 1.)):
                    raise ValueError('uniform real/zero padding population required')
                if not slot['padding']:seen[slot['id']]+=1
        if seen!=Counter(population.keys()):raise ValueError('full epoch coverage differs')
        all_batches.extend(batches)
    if len(all_batches)!=manifest['total_updates_per_stage']:raise ValueError('stage cursor bound differs')
    return all_batches,manifest


def build_epoch(rows,lengths,*,epoch,seed=317,global_batch=128,world_size=4,microbatch=8):
    if (not rows or len({r['id'] for r in rows})!=len(rows) or
            set(lengths)!={r['id'] for r in rows} or type(epoch) is not int or epoch<0 or
            any(type(x) is not int or x<1 for x in (global_batch,world_size,microbatch)) or
            global_batch%(world_size*microbatch)):
        raise ValueError('complete distinct population, lengths and divisible DP layout required')
    if any(r['role']!='learning' or r['known_pending_source'] or r['official_split']!='train' for r in rows):
        raise ValueError('only admitted learning TRAIN records may optimize')
    shuffled=sorted(rows,key=lambda r:hashlib.sha256(repr((seed,epoch,r['id'])).encode()).digest())
    order=class_spread_indices([r['intent'] for r in shuffled],seed=seed,epoch=epoch)
    real=[dict(id=shuffled[i]['id'],weight=1.,padding=False) for i in order]
    slots=list(real)
    while len(slots)%global_batch:slots.append(dict(real[(len(slots)-len(real))%len(real)],weight=0.,padding=True))
    batches=[]
    for index,start in enumerate(range(0,len(slots),global_batch)):
        group=slots[start:start+global_batch]
        ranks=length_group_indices([lengths[s['id']] for s in group],microbatch=microbatch,
            world_size=world_size,order='length_grouped')
        batches.append(dict(epoch=epoch,index=index,slots=group,rank_indices=ranks))
    # Independently enumerate all rank assignments and retain real denominators.
    actual=Counter()
    for batch in batches:
        if sorted(i for indices in batch['rank_indices'] for i in indices)!=list(range(global_batch)):
            raise ValueError('rank allocation is not a permutation')
        actual.update(s['id'] for s in batch['slots'] if not s['padding'])
    if actual!=Counter(r['id'] for r in rows):raise ValueError('population coverage differs')
    return dict(batches=batches,report=dict(epoch=epoch,recordings=len(rows),updates=len(batches),
        padding_slots=len(slots)-len(rows),source_groups=len({r['source_group'] for r in rows}),
        class_counts=dict(sorted(Counter(r['intent'] for r in rows).items())),
        global_batch=global_batch,world_size=world_size,microbatch=microbatch,
        accumulation=global_batch//world_size//microbatch,
        weights='uniform_real_recordings_zero_padding',batches_sha256=content_hash(batches)))
