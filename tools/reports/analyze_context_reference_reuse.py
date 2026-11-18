"""Read-only full-schedule reference reuse accounting from audited input hashes.

No cache mutation or runtime dispatch. Tensor fingerprints identify equality;
they do not replace ReferenceCache's exact filename key using marker values.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from tools.jev.context_cache_receipts import parse_stream


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def count_reuse(batches, identities, *, initial=None):
    seen=set(initial or ());steps=[]
    for step,batch in enumerate(batches,1):
        keys=[];real=padding=0
        for slot in batch['slots']:
            if (type(slot['padding']) is not bool or slot['weight']!=(0. if slot['padding'] else 1.)
                    or slot['id'] not in identities):raise ValueError('Known input and real/padding weights required')
            keys.append(identities[slot['id']]);padding+=slot['padding'];real+=not slot['padding']
        unique=set(keys);new=unique-seen
        steps.append(dict(step=step,epoch=batch['epoch'],slots=len(keys),real=real,padding=padding,
            distinct_inputs=len(unique),new_reference_forwards=len(new),
            prior_cache_slots=sum(k in seen for k in keys),same_batch_duplicate_slots=len(keys)-len(unique)))
        seen.update(unique)
    return steps,seen


def run(schedule,receipts,cache_proof,grpo_config,rloo_config,timing):
    schedule=Path(schedule);m=read(schedule/'MANIFEST.json');proof=read(cache_proof)
    if (sha(receipts)!=proof['independent_receipts_sha256'] or proof['state']['ExitCode']!=0
            or proof['state']['Status']!='exited' or proof['state']['OOMKilled']):
        raise ValueError('Successful independently audited full input receipts required')
    stream,_=parse_stream(Path(receipts).read_text())
    records=[r['value'] for r in stream[1:-1]]
    if (len(records)!=40513 or len({r['id'] for r in records})!=40513
            or proof['manifest']['recordings']!=40513
            or proof['manifest']['release_sha256']!=m['release_sha256']):
        raise ValueError('Complete same learning population required')
    grpo,rloo=read(grpo_config),read(rloo_config)
    same_fields=('source_policy_id','source_contract','base_id','schedule_sha256','encoded_audio_cache')
    if any(grpo[k]!=rloo[k] for k in same_fields):raise ValueError('Reference/input setup differs across algorithms')
    if grpo['schedule_sha256']!=sha(schedule/'MANIFEST.json'):raise ValueError('Frozen schedule differs')
    if grpo['algorithm']!='grpo_v1' or rloo['algorithm']!='rloo_v1':raise ValueError('Matched algorithm pair required')
    identities={}
    for r in records:
        if (set(r['tensors'])!={'input_ids','attention_mask','input_features','input_features_mask','positions'}
                or r['tensors']['positions']['shape']!=[1,60]):raise ValueError('Full60 input tensors required')
        identities[r['id']]=digest(dict(policy_id=grpo['source_policy_id'],tensors=r['tensors']))
    batches=[];epoch_ids=[]
    for epoch,entry in enumerate(m['epochs']):
        path=schedule/f'epoch-{epoch}.json'
        if sha(path)!=entry['file_sha256']:raise ValueError('Epoch source changed')
        data=read(path);bs=data['batches']
        real=[slot['id'] for b in bs for slot in b['slots'] if not slot['padding']]
        if Counter(real)!=Counter(identities.keys()) or len(bs)!=317:
            raise ValueError('Each epoch must cover all40513 exactly once')
        epoch_ids.append(real);batches.extend(bs)
    if len(batches)!=634:raise ValueError('Full two epoch budget required')
    steps,seen=count_reuse(batches,identities)
    second,shared=count_reuse(batches,identities,initial=seen)
    separate_roots=grpo['reference_cache']!=rloo['reference_cache']
    summary=[]
    for epoch in range(2):
        entries=[s for s in steps if s['epoch']==epoch]
        summary.append(dict(epoch=epoch,updates=len(entries),**{k:sum(s[k] for s in entries)
            for k in ('slots','real','padding','new_reference_forwards','prior_cache_slots','same_batch_duplicate_slots')}))
    timing_data=read(timing)
    cost_per_miss=timing_data['seconds']['reference_seconds']*timing_data['steps']/timing_data['reference_misses']
    n=len(seen);per_run=sum(x['new_reference_forwards'] for x in steps)
    saved=per_run-sum(x['new_reference_forwards'] for x in second)
    # This rate includes switching/cache writes and was measured only in an early
    # prefix. It is a planning estimate, not an observed steady-state speedup.
    result=dict(schema='context-reference-full-schedule-reuse-analysis-v1',
        inputs=40513,distinct_encoded_inputs=n,updates=634,epochs=summary,
        tensor_fingerprint_scope='Equality of all audited tensor bytes/dtypes/shapes including marker positions; not literal cache filename.',
        fixed_reference_policy_id=grpo['source_policy_id'],same_reference_and_input_configuration=True,
        configured_cache_roots=dict(grpo=grpo['reference_cache'],rloo=rloo['reference_cache']),
        configured_separate_roots=separate_roots,
        expected_cold_reference_forwards_per_algorithm=per_run,
        expected_separate_two_algorithm_reference_forwards=2*per_run,
        candidate_shared_two_algorithm_reference_forwards=per_run+sum(x['new_reference_forwards'] for x in second),
        candidate_reference_forwards_saved=saved,
        raw_probability_storage_bytes=n*60*4,
        raw_probability_storage_scope='FP32 probability payload only; excludes JSON, contracts, hashes, filesystem and read overhead.',
        planning_only=dict(prefix_seconds_per_new_reference=cost_per_miss,
            potential_reference_seconds_saved=saved*cost_per_miss,
            limitation='Prefix-rate extrapolation only. No cache sharing executed; no measured end-to-end speedup. Existing second-epoch reuse already counted.'),
        prerequisites=['same fixed policy and actual input/positions/runtime numerics',
            'independent source cache provenance and full digest verification',
            'immutable local cache access and separate optimizer/behavior states',
            'new version, matched compute reporting, update/parity checks before training'],
        limitations=['Current cache namespace binds policy/base/contract, but not a separate full runtime source digest.',
            'Actual second-epoch cache hits and RLOO misses require later runtime observations.',
            'Current and queued packages remain unchanged; no reference forward is skipped here.'],
        sources={str(p):sha(p) for p in (schedule/'MANIFEST.json',schedule/'epoch-0.json',schedule/'epoch-1.json',
            receipts,cache_proof,grpo_config,rloo_config,timing,
            'tools/jev/context_reference_cache.py','tools/jev/context_rollout_session.py',
            'src/multimodal_jev/cached_context_batches.py')},steps=steps)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('schedule','receipts','cache-proof','grpo-config','rloo-config','timing','output'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args();d=run(a.schedule,a.receipts,a.cache_proof,a.grpo_config,a.rloo_config,a.timing)
    with Path(a.output).open('x') as f:json.dump(d,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in d.items() if k not in ('sources','steps')},indent=2))
