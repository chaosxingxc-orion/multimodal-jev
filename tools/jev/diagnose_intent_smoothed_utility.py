#!/usr/bin/env python3
"""Offline original-vs-utility logit credit on accepted single-Final policies."""
import argparse
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.slurp_intent_pilot import load_pilot
from multimodal_jev.intent_actor_inputs import schedule
from multimodal_jev.span_credit_concentration import sensitivity,concentration
from multimodal_jev.span_smoothed_utility import utility_credit
from queue_span_native_actor_npu import read,completed
from queue_span_dp4_resume import check_pins
from queue_intent_finite_npu import verify_gate


def run(config):
    check_pins(config['pins']);items,_=load_pilot(config['pilot'],config['manifest_sha256'],'D')
    reports=[];evidence=dict(config['pins'])
    for entry in config['gates']:
        root=Path(entry['root']);launch=read(root/'launch.json')
        if launch['image']!=config['image'] or not completed(launch['container_id'],config['image']):raise ValueError('actual accepted policy required')
        gate=verify_gate(root,config['pilot'],config['manifest_sha256']);data=read(root/'run/vllm/result.json');records={r['id']:r for r in data['records']};rows={}
        for item in items:
            p=records[item['id']+'::D']['candidate_logprobs'][0];target=item['target'];y=target['targets'][0]
            s=sensitivity(p,y,target['request_weight']);u=utility_credit(s['gold_probability'],len(p))
            rows[item['id']]=dict(id=item['id'],gold_probability=s['gold_probability'],original_norm=s['norm'],utility_norm=s['norm']*u['scale'],**u)
        groups={}
        for name,population in [('all_learning',items),*[(f'schedule_batch_{i+1:02d}',schedule(items,317,i,8)) for i in range(12)]]:
            selected=[rows[i['id']] for i in population]
            groups[name]={variant:concentration([dict(id=r['id'],gold_probability=r['gold_probability'],norm=r[variant+'_norm']) for r in selected]) for variant in ('original','utility')}
            groups[name]['scale_range']=[min(r['scale'] for r in selected),max(r['scale'] for r in selected)]
        reports.append(dict(update=entry['update'],adapter_id=data['adapter_id'],gate=gate,groups=groups,records=list(rows.values())))
        for name in ('config.json','launch.json','run/READY.json','run/vllm/result.json','run/hf/result.json'):evidence[str(root/name)]=file_hash(root/name)
    for p in (Path(__file__),Path(__file__).resolve().parents[2]/'src/multimodal_jev/span_smoothed_utility.py'):evidence[str(p)]=file_hash(p)
    return dict(reports=reports,files=evidence,scope='Fixed-checkpoint counterfactual12batch credit, not actual trajectory. Independent output-logit norms, not shared gradients or task gain. Bounded log utility changes optimization target; no NPU training.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable utility diagnosis exists')
    atomic_json(a.output,run(read(a.config)))
