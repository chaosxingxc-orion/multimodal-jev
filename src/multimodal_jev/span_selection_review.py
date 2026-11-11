"""Deterministic review sampling from the reserved source-selection role.

Native entity presence is a sampling stratum, never an approved negative or
reward. Reference text belongs in the offline review packet, not actor input.
"""
from collections import Counter, defaultdict
import hashlib
import itertools


def select_review_packet(requests, targets, metadata, learning_requests, per_stratum=6, seed=17):
    if type(per_stratum) is not int or per_stratum < 1 or type(seed) is not int:
        raise ValueError('positive review budget and integer seed required')
    by_id={r['id']:r for r in requests}; gold={t['id']:t for t in targets}
    if len(by_id)!=len(requests) or len(gold)!=len(targets) or by_id.keys()!=gold.keys():
        raise ValueError('complete unique source request/target join required')
    learned_groups={r['source_group'] for r in learning_requests}
    learned_audio={r['audio_sha256'] for r in learning_requests}
    if any(r['role']!='learning' for r in learning_requests):
        raise ValueError('explicit learning population required')
    by_source={str(m['slurp_id']):m for m in metadata}
    if len(by_source)!=len(metadata):raise ValueError('unique clean metadata source IDs required')
    groups=defaultdict(list)
    for uid,request in by_id.items():
        target=gold[uid]
        if (request['role']!='selection' or request['official_split']!='train'
                or request['source_group'] in learned_groups or request['audio_sha256'] in learned_audio):
            raise ValueError('reserved selection source/audio overlaps learning or has wrong role')
        if target['annotation_status']!='available' or target['optimization_allowed'] is not False:
            raise ValueError('raw source must remain unavailable for optimization')
        types={e['type'] for e in target['native_target']['entities']}
        signature=''.join(str(int(k in types)) for k in ('date','time','timeofday'))
        sources=[by_source[k] for k in target['source_ids']]
        references=sorted({s['sentence'] for s in sources})
        if len(references)!=1:raise ValueError('ambiguous source reference')
        if any(request['id'].removeprefix('slurp:') not in {r['file'] for r in s['recordings']} for s in sources):
            raise ValueError('recording is not attached to declared source')
        groups[request['source_group']].append(dict(request=request,target=target,
            native_sampling_stratum=signature,reference=references[0],source_ids=target['source_ids'],
            review_status={k:'pending' for k in ('date','time','timeofday')},
            optimization_allowed=False,evaluation_admitted=False))
    buckets=defaultdict(list)
    def rank(value):return hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()
    for source,items in groups.items():
        if len({r['native_sampling_stratum'] for r in items})!=1:
            raise ValueError('conflicting sampling stratum within source')
        chosen=min(items,key=lambda r:rank(r['request']['id']))
        buckets[chosen['native_sampling_stratum']].append(chosen)
    selected=[];counts={}
    for bits in itertools.product('01',repeat=3):
        signature=''.join(bits)
        pool=sorted(buckets[signature],key=lambda r:rank(r['request']['source_group']))
        selected.extend(pool[:per_stratum])
        counts[signature]=dict(available_source_groups=len(pool),selected=min(per_stratum,len(pool)),
                               shortfall=max(0,per_stratum-len(pool)))
    return selected,dict(seed=seed,requested_per_native_stratum=per_stratum,strata=counts,
        selection_recordings=len(requests),selection_source_groups=len(groups),
        selected_recordings=len(selected),source_overlap_with_learning=0,audio_hash_overlap_with_learning=0,
        training_admitted=False,evaluation_admitted=False,test_opened=False,
        speaker_disjoint=False,semantic_near_duplicate_isolation=False,
        scope='Stratified pending semantic review of reserved selection sources; native absence is not verified false. Not representative official-test accuracy or unseen-rule evidence.')
