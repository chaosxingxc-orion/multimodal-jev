#!/usr/bin/env python3
"""Counterfactual compatibility penalty on complete accepted learning gates."""
import argparse
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.slurp_intent_pilot import load_pilot
from multimodal_jev.intent_relation_credit import credit
from multimodal_jev.span_credit_concentration import sensitivity
from queue_span_native_actor_npu import read,completed
from queue_span_dp4_resume import check_pins
from queue_intent_finite_npu import verify_gate


def run(config):
    check_pins(config['pins']);items,_=load_pilot(config['pilot'],config['manifest_sha256'],'P')
    reports=[];evidence=dict(config['pins'])
    for entry in config['gates']:
        root=Path(entry['root']);launch=read(root/'launch.json')
        if launch['image']!=config['image'] or not completed(launch['container_id'],config['image']):
            raise ValueError('actual complete gate required')
        audit=verify_gate(root,config['pilot'],config['manifest_sha256']);data=read(root/'run/vllm/result.json')
        records={r['id']:r for r in data['records']};rows=[]
        for item in items:
            q=item['request'].questions;s,a,f=[x.values for x in q]
            pairs=[]
            for value in f:
                matches=[(i,j) for i,x in enumerate(s) for j,y in enumerate(a) if x+'_'+y==value]
                if len(matches)!=1:raise ValueError('unambiguous canonical relation required')
                pairs.append(matches[0])
            logp=records[item['id']+'::Q']['candidate_logprobs'];r=credit(logp,pairs,.1)
            final=sensitivity(logp[-1],item['target']['targets'][-1])['gradient']
            relation=r['logit_gradients'][-1]
            dot=sum(x*y for x,y in zip(final,relation,strict=True))
            rows.append(dict(id=item['id'],request_weight=item['target']['request_weight'],
                             relation_final_correctness_logit_dot=dot,**r))
        reports.append(dict(update=entry['update'],adapter_id=data['adapter_id'],gate=audit,records=rows,
            recordings=len(rows),relation_opposes_final_correctness=sum(r['relation_final_correctness_logit_dot']<0 for r in rows),
            mean_expected_compatibility=sum(r['expected_compatibility'] for r in rows)/len(rows),
            mean_expected_penalty=sum(r['expected_penalty'] for r in rows)/len(rows)))
        for name in ('config.json','launch.json','run/READY.json','run/vllm/result.json','run/hf/result.json'):
            evidence[str(root/name)]=file_hash(root/name)
    for p in (Path(__file__),Path(__file__).resolve().parents[2]/'src/multimodal_jev/intent_relation_credit.py'):
        evidence[str(p)]=file_hash(p)
    return dict(reports=reports,files=evidence,scope='Counterfactual -0.1 incompatibility penalty, no training. Per-source Final output-logit derivative dot product only; not shared parameter gradients or efficacy. Update labels have external curve lineage.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable diagnosis exists')
    atomic_json(a.output,run(read(a.config)))
