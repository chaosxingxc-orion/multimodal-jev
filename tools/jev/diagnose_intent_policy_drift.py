#!/usr/bin/env python3
"""Compare full accepted intent policies, including post-update surrogate drift."""
import argparse
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.slurp_intent_pilot import load_pilot
from multimodal_jev.span_policy_drift import diagnose
from queue_span_native_actor_npu import read,completed
from queue_intent_finite_npu import verify_gate,verify_final


def run(training,suite):
    training,suite=Path(training),Path(suite);cfg=read(training/'config.json');previous=Path(cfg['acceptance_root'])
    for root in (training,previous,suite):
        launch=read(root/'launch.json')
        if not completed(launch['container_id'],launch['image']):raise ValueError('actual completed training/gates required')
    verify_final(suite)
    accepted=read(suite/'verified.json')
    if accepted['root']!=str(training) or accepted['updates']!=cfg['step']+1:raise ValueError('update lineage differs')
    gates=[verify_gate(p,cfg['pilot'],cfg['manifest_sha256']) for p in (previous,suite)]
    records=[{r['id']:r for r in read(p/'run/vllm/result.json')['records']} for p in (previous,suite)]
    selected=set(read(training/'run/selection.json')['recording_ids']);reports={}
    for arm in ('D','Q'):
        items,_=load_pilot(cfg['pilot'],cfg['manifest_sha256'],arm);rows=[]
        for item in items:
            uid=item['id']+'::'+arm;old,new=[r[uid]['candidate_logprobs'] for r in records]
            rows.append(dict(id=item['id'],request_weight=item['target']['request_weight'],
                spans={q.id:diagnose(a,b,y) for q,a,b,y in zip(item['request'].questions,old,new,item['target']['targets'],strict=True)}))
        groups={}
        for name,subset in (('all_learning',rows),('updated_this_step',[r for r in rows if r['id'] in selected]),
                            ('other_learning',[r for r in rows if r['id'] not in selected])):
            weight=sum(r['request_weight'] for r in subset);spans={}
            for span in rows[0]['spans']:
                values=[r['spans'][span] for r in subset]
                spans[span]=dict(mean={k:sum(v[k] for v in values)/len(values) for k in values[0]},
                    max_kl_old_to_new=max(v['kl_old_to_new'] for v in values),
                    max_abs_log_ratio=max(v['maximum_abs_log_ratio'] for v in values),
                    request_weighted={k:sum(r['request_weight']*r['spans'][span][k] for r in subset)/weight
                                      for k in ('gold_probability_delta','postupdate_clipped_surrogate','postupdate_dual_clipped_surrogate','kl_old_to_new')})
            groups[name]=dict(recordings=len(subset),spans=spans)
        reports[arm]=dict(groups=groups,records=rows)
    paths=[training/'config.json',training/'run/selection.json',suite/'verified.json',suite/'ACCEPTED.json',
           previous/'run/vllm/result.json',suite/'run/vllm/result.json',Path(__file__).resolve(),
           Path(__file__).resolve().parents[2]/'src/multimodal_jev/span_policy_drift.py']
    return dict(update=cfg['step']+1,training_arm=cfg['arm'],reports=reports,gate_audits=gates,
        files={str(p):file_hash(p) for p in paths},
        scope='Post-update learning-population finite policy drift. Not measured training-time clip fraction, '
              'trust-region guarantee, parameter-gradient attribution or held-out efficacy. Q is inference on D-trained weights when training_arm=D.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('training','suite','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError('immutable drift report exists')
    atomic_json(out,run(a.training,a.suite))
