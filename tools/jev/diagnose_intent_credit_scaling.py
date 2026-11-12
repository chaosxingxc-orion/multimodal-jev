#!/usr/bin/env python3
"""Fixed, source-only counterfactual credit weighting; no model update."""
import argparse
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.slurp_intent_pilot import load_pilot
from multimodal_jev.intent_actor_inputs import schedule
from multimodal_jev.span_credit_concentration import sensitivity,concentration
from multimodal_jev.span_credit_scaling import bounded_scales
from queue_span_native_actor_npu import read,completed
from queue_intent_finite_npu import verify_gate


def run(config):
    for p,h in config['pins'].items():
        if file_hash(p)!=h:raise ValueError('diagnostic input changed')
    items,_=load_pilot(config['pilot'],config['manifest_sha256'],'D')
    results=[]
    for entry in config['gates']:
        gate=Path(entry['root']);launch=read(gate/'launch.json')
        if launch['image']!=config['image'] or not completed(launch['container_id'],config['image']):
            raise ValueError('actual completed fixed-image gate required')
        verify_gate(gate,config['pilot'],config['manifest_sha256'])
        raw=read(gate/'run/vllm/result.json');records={r['id']:r for r in raw['records']}
        groups={'all_learning':items,**{f'schedule_batch_{i+1:02d}':schedule(items,317,i,8) for i in range(12)}}
        reports={}
        for name,selected in groups.items():
            rows=[];supports=[];weights=[]
            for item in selected:
                target=item['target']
                if target['supervised']!=[True] or target['span_weights']!=[1.]:raise ValueError('reviewed single Final only')
                logp=records[item['id']+'::D']['candidate_logprobs'][0];weight=target['request_weight']
                info=sensitivity(logp,target['targets'][0],weight)
                rows.append(dict(id=item['id'],gold_probability=info['gold_probability'],norm=info['norm']))
                supports.append(len(logp));weights.append(weight)
            scaling=bounded_scales([r['gold_probability'] for r in rows],supports,weights)
            denominator=sum(weights)
            before=[dict(r,norm=r['norm']/denominator) for r in rows]
            after=[dict(r,norm=r['norm']*s/denominator) for r,s in zip(rows,scaling['scales'],strict=True)]
            reports[name]=dict(before=concentration(before),after=concentration(after),
                scales=scaling,below_floor=sum(r['gold_probability']<1/k for r,k in zip(rows,supports,strict=True)),
                scope='Independent-logit credit only. Population normalization and each scheduled batch normalization are distinct.')
        results.append(dict(update=entry['update'],adapter_id=raw['adapter_id'],groups=reports))
    return dict(results=results,scope='Prospective bounded inverse-gold-probability source weighting, fixed floor1/K and weighted-mean normalization. '
        'Learning data only, no weight update, no shared-parameter attribution, no algorithm efficacy or generalization result.',
        files=config['pins'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable diagnosis exists')
    atomic_json(a.output,run(read(a.config)))
