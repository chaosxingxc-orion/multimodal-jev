#!/usr/bin/env python3
"""Relate accepted batch credit to observed source coverage, without training."""
import argparse
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.slurp_intent_pilot import load_pilot
from multimodal_jev.span_credit_concentration import sensitivity,concentration
from diagnose_intent_policy_drift import run as drift
from queue_span_native_actor_npu import read


def run(training,suite):
    training,suite=Path(training),Path(suite)
    diagnosis=drift(training,suite);cfg=read(training/'config.json')
    old=Path(cfg['acceptance_root']);records={r['id']:r for r in read(old/'run/vllm/result.json')['records']}
    selected=set(read(training/'run/selection.json')['recording_ids'])
    arm=cfg['arm'];wire='D' if arm=='D' else 'Q'
    items,_=load_pilot(cfg['pilot'],cfg['manifest_sha256'],arm)
    batch=[i for i in items if i['id'] in selected]
    if len(batch)!=cfg['global_requests'] or len(batch)!=len(selected):raise ValueError('exact actual batch required')
    denominator=sum(i['target']['request_weight'] for i in batch)
    observed={r['id']:r for r in diagnosis['reports'][wire]['records']}
    spans={}
    for j,question in enumerate(batch[0]['request'].questions):
        rows=[]
        for item in batch:
            target=item['target'];weight=target['request_weight']*target['span_weights'][j]/denominator
            s=sensitivity(records[item['id']+'::'+wire]['candidate_logprobs'][j],target['targets'][j],weight)
            change=observed[item['id']]['spans'][question.id]['gold_probability_delta']
            rows.append(dict(id=item['id'],gold=target['target_values'][j],gold_probability=s['gold_probability'],
                norm=s['norm'],request_weight=target['request_weight'],span_weight=target['span_weights'][j],
                gold_probability_delta=change))
        spans[question.id]=dict(logit_credit=concentration(rows),rows=rows)
    populations={}
    final=batch[0]['request'].final_id
    for name,subset in [('updated',[r for r in observed.values() if r['id'] in selected]),
                        ('other_learning',[r for r in observed.values() if r['id'] not in selected]),
                        ('all_learning',list(observed.values()))]:
        deltas=[r['spans'][final]['gold_probability_delta'] for r in subset]
        ordered=sorted(deltas);n=len(ordered)
        populations[name]=dict(recordings=n,increased=sum(d>0 for d in deltas),decreased=sum(d<0 for d in deltas),
            unchanged=sum(d==0 for d in deltas),mean_delta=sum(deltas)/n if n else None,
            median_delta=(ordered[n//2] if n%2 else (ordered[n//2-1]+ordered[n//2])/2) if n else None)
    files=dict(diagnosis['files'])
    for path in (Path(__file__).resolve(),Path(__file__).resolve().parents[2]/'src/multimodal_jev/span_credit_concentration.py'):
        files[str(path)]=file_hash(path)
    return dict(update=diagnosis['update'],training_arm=arm,spans=spans,populations=populations,files=files,
        scope='Learning-only actual batch credit and before/after probability changes. Independent-logit norms are not shared-parameter gradient contributions or causal attribution.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--training',required=True);p.add_argument('--suite',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable coverage exists')
    atomic_json(a.output,run(a.training,a.suite))
