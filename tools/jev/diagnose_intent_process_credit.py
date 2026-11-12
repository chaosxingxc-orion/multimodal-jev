#!/usr/bin/env python3
"""Counterfactual P credit on accepted Q-template policies; no P training claim."""
import argparse
from pathlib import Path
from statistics import median
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.slurp_intent_pilot import load_pilot
from multimodal_jev.intent_actor_inputs import schedule
from multimodal_jev.span_process_credit_budget import measure
from queue_span_native_actor_npu import read,completed
from queue_intent_finite_npu import verify_gate


def run(config):
    for p,h in config['pins'].items():
        if file_hash(p)!=h:raise ValueError('pinned diagnostic evidence changed')
    items,_=load_pilot(config['pilot'],config['manifest_sha256'],'P');reports=[];evidence=dict(config['pins'])
    for entry in config['gates']:
        root=Path(entry['root']);launch=read(root/'launch.json')
        if launch['image']!=config['image'] or not completed(launch['container_id'],config['image']):
            raise ValueError('actual complete policy required')
        audit=verify_gate(root,config['pilot'],config['manifest_sha256'])
        result=read(root/'run/vllm/result.json');records={r['id']:r for r in result['records']};rows=[]
        for item in items:
            target=item['target']
            if target['supervised']!=[True]*3 or target['span_weights']!=[.1,.1,1.]:
                raise ValueError('fixed reviewed P credit required')
            rows.append(dict(id=item['id'],request_weight=target['request_weight'],
                **measure(records[item['id']+'::Q']['candidate_logprobs'],target['targets'],target['span_weights'])))
        indexed={r['id']:r for r in rows}
        groups={}
        for name,subset in [('all_learning',items),*[(f'schedule_batch_{i+1:02d}',schedule(items,317,i,8)) for i in range(12)]]:
            chosen=[indexed[i['id']] for i in subset]
            # Separate output coordinates for every request and Span.
            process=sum((r['request_weight']*r['process_norm'])**2 for r in chosen)
            final=sum((r['request_weight']*r['final_norm'])**2 for r in chosen)
            ratios=[r['process_to_final_norm'] for r in chosen if r['process_to_final_norm'] is not None]
            groups[name]=dict(recordings=len(chosen),process_exceeds_final=sum(r['process_exceeds_final'] for r in chosen),
                median_per_request_norm_ratio=median(ratios) if ratios else None,
                weighted_disjoint_process_to_final_norm=(process/final)**.5 if final else None,
                weighted_disjoint_process_squared_share=process/(process+final) if process+final else None)
        reports.append(dict(update=entry['update'],adapter_id=result['adapter_id'],gate=audit,groups=groups,records=rows))
        for name in ('config.json','launch.json','run/READY.json','run/vllm/result.json','run/hf/result.json'):
            evidence[str(root/name)]=file_hash(root/name)
    for p in (Path(__file__),Path(__file__).resolve().parents[2]/'src/multimodal_jev/span_process_credit_budget.py'):
        evidence[str(p)]=file_hash(p)
    return dict(reports=reports,files=evidence,scope='Counterfactual P additive credit on fixed accepted Q-template outputs of D policies. Disjoint output-logit coordinates only, not shared-parameter gradients, P training results or process efficacy.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable diagnostic exists')
    atomic_json(a.output,run(read(a.config)))
