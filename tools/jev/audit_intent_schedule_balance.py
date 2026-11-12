#!/usr/bin/env python3
"""Read the original matched schedule and audit its effective class coefficients."""
import argparse
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.intent_schedule_balance import audit
from multimodal_jev.intent_actor_inputs import schedule
from multimodal_jev.slurp_intent_pilot import load_pilot
from queue_span_native_actor_npu import read


def run(config_path,protocol_path):
    config_path,protocol_path=Path(config_path),Path(protocol_path)
    cfg=read(config_path);protocol=read(protocol_path)
    items,admission=load_pilot(cfg['pilot'],cfg['manifest_sha256'],'D')
    if (protocol['seed']!=cfg['seed'] or protocol['global_requests_per_update']!=cfg['global_requests']
            or protocol['total_updates']!=12):
        raise ValueError('fixed original matched schedule required')
    actual=[[r['id'] for r in schedule(items,cfg['seed'],step,cfg['global_requests'])] for step in range(12)]
    if actual!=protocol['schedule']:raise ValueError('registered schedule differs from actor schedule')
    sources=[dict(id=i['id'],label=i['request'].questions[-1].values[i['target']['targets'][-1]],
                  weight=i['target']['request_weight']) for i in items]
    for path,digest in cfg['pins'].items():
        if file_hash(path)!=digest:raise ValueError('training evidence changed')
    result=audit(sources,actual)
    # Same source exposures, 32-way accumulation is a separately registered control.
    grouped=[sum(actual[i:i+4],[]) for i in range(0,12,4)]
    result['same_exposures_batch32_counterfactual']=audit(sources,grouped)
    paths=[config_path,protocol_path,Path(cfg['pilot'])/'MANIFEST.json',Path(__file__).resolve(),
           Path(__file__).resolve().parents[2]/'src/multimodal_jev/intent_schedule_balance.py',
           Path(__file__).resolve().parents[2]/'src/multimodal_jev/intent_actor_inputs.py']
    return dict(result,admission=admission,files={str(p):file_hash(p) for p in paths})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('config','protocol','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable audit exists')
    atomic_json(a.output,run(a.config,a.protocol))
