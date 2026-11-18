"""TRAIN-only live diagnostic with frozen schedule/audio/tensor/gold binding."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct

from tools.reports.analyze_context_action_diversity import analyze as action_audit
from tools.jev.context_reward_signal import singleton_signal
from tools.jev.context_cache_receipts import parse_stream


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def decision(prob,gold):
    chosen=[i for i,p in enumerate(prob) if p>=.5]
    if any(p in (0,1) for p in prob):raise ValueError('Endpoint probabilities need explicit infinite-NLL handling')
    return dict(top1_correct=max(range(len(prob)),key=prob.__getitem__)==gold,
        exact_set=chosen==[gold],set_f1=2/(len(chosen)+1) if gold in chosen else 0.,
        selected_count=len(chosen),binary_nll=-sum(math.log(p) if i==gold else math.log1p(-p) for i,p in enumerate(prob))/len(prob),
        brier=sum((p-(i==gold))**2 for i,p in enumerate(prob))/len(prob))


def summarize(rows,side):
    result={'queries':len(rows)}
    if not rows:return result
    for block in ('decision','signal'):
        result[block]={k:sum(r[side][block][k] for r in rows)/len(rows) for k in rows[0][side][block] if k not in ('group_size','candidates')}
    return result


def run(observation,release,schedule,receipts,cache_proof):
    preliminary=action_audit(observation);live=read(observation);release=Path(release);schedule=Path(schedule)
    rm=read(release/'MANIFEST.json');sm=read(schedule/'MANIFEST.json');proof=read(cache_proof)
    if (sha(release/'MANIFEST.json')!='38e4273d9328e4f02db081315e46f97a17628b974be7ba175254882e95421482'
            or sha(schedule/'MANIFEST.json')!='d9cd5832aa160df46a4c10a2051ff12c352d2a2609e2cffd6901267b28a814f6'
            or sm['release_sha256']!=sha(release/'MANIFEST.json')):raise ValueError('Frozen release/schedule changed')
    for name in ('learning.jsonl','tasks.json'):
        if sha(release/name)!=rm['files'][name]:raise ValueError('Release asset changed')
    if (sha(receipts)!=proof['independent_receipts_sha256'] or proof['state']['ExitCode']!=0
            or proof['state']['OOMKilled'] or proof['manifest']['release_sha256']!=sm['release_sha256']):raise ValueError('Cache evidence changed')
    learning=[json.loads(x) for x in (release/'learning.jsonl').read_text().splitlines()]
    labels={r['id']:r for r in learning}
    tensor_rows,_=parse_stream(Path(receipts).read_text())
    tensors={r['value']['id']:r['value'] for r in tensor_rows if r['kind']=='input'}
    if len(labels)!=40513 or set(tensors)!=set(labels):raise ValueError('Full learning population required')
    candidates=sorted(read(release/'tasks.json')['intent']['candidate_pairs'])
    if len(candidates)!=60:raise ValueError('Full candidate menu required')
    update=preliminary['behavior_update'];epoch,batch=divmod(update,317);ep=schedule/f'epoch-{epoch}.json'
    if sha(ep)!=sm['epochs'][epoch]['file_sha256']:raise ValueError('Epoch changed')
    slots=read(ep)['batches'][batch]['slots'];queries=live['progress']['latest'][0]['data']['rollout']['queries'];details=[]
    for index,(slot,q) in enumerate(zip(slots,queries)):
        uid=slot['id'];gold=candidates.index(labels[uid]['intent']);native=labels[uid];cache=tensors[uid]
        if (slot['padding'] or slot['weight']!=1 or native['role']!='learning' or native['official_split']!='train'
                or native['known_pending_source'] or q['request_id']!=f'{uid}::epoch{epoch}-batch{batch}-slot{index}'
                or cache['audio_sha256']!=native['audio_sha256']):raise ValueError('Real TRAIN/audio/schedule binding mismatch')
        identity=q['behavior_record']['request_identity']
        if identity['inputs']!={k:v for k,v in cache['tensors'].items() if k!='positions'}:raise ValueError('Encoded audio/input binding mismatch')
        if hashlib.sha256(struct.pack('<60q',*identity['positions'])).hexdigest()!=cache['tensors']['positions']['sha256']:
            raise ValueError('Candidate marker ordering mismatch')
        d={'id':uid,'source_group':native['source_group'],'gold_index':gold}
        for side in ('behavior','reference'):
            prob=q[side+'_record']['prob'];d[side]={'decision':decision(prob,gold),'signal':singleton_signal(prob,gold)}
        details.append(d)
    if len(slots)!=128 or len(details)!=128:raise ValueError('Full batch required')
    strata={}
    for side in ('behavior','reference'):
        strata[side]={}
        for correct in (True,False):
            for saturated in (True,False):
                subset=[r for r in details if r[side]['decision']['exact_set']==correct and (r[side]['signal']['group_all_equal_reward_probability']>.95)==saturated]
                strata[side][f'exact_{correct}_reward_collision_over95_{saturated}']=summarize(subset,side)
    sources={str(p):sha(p) for p in (Path(observation),release/'MANIFEST.json',release/'learning.jsonl',release/'tasks.json',schedule/'MANIFEST.json',ep,Path(receipts),Path(cache_proof),Path(__file__),Path('tools/jev/context_reward_signal.py'),Path('tools/reports/analyze_context_action_diversity.py'))}
    return {'schema':'context-live-reward-support-v1','sources':sources,'actor_update':update+1,'behavior_update':update,
        'queries':details,'source_groups':len({r['source_group'] for r in details}),
        'summaries':{s:summarize(details,s) for s in ('behavior','reference')},'strata':strata,
        'paired_exact_changes':dict(Counter(f"reference_{r['reference']['decision']['exact_set']}_behavior_{r['behavior']['decision']['exact_set']}" for r in details)),
        'scope':'Single live TRAIN batch; exact sampling distribution and frozen-threshold diagnostics, not heldout efficacy or actual action replay.',
        'limits':['Full run not independently audited yet.','This batch may already have trained reference; not a generalization estimate.',
                  'No causal attribution between RL, auxiliary supervision, and KL.','No configuration changes or new admission requirements.']}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('observation','release','schedule','receipts','cache-proof','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();r=run(a.observation,a.release,a.schedule,a.receipts,a.cache_proof)
    with Path(a.output).open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps({k:r[k] for k in ('summaries','strata','paired_exact_changes')},indent=2))
