#!/usr/bin/env python3
"""Exercise replay comparison on real accepted DP4 shards and corrupt copies.

Self-comparison validates layout only, not the unexecuted residency replay.
"""
import argparse
import copy
from pathlib import Path
import torch
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.intent_residency_comparison import compare_states
from queue_span_native_actor_npu import read


def run(config):
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU-only probe required')
    for path,h in config['pins'].items():
        if file_hash(path)!=h:raise ValueError('pinned input changed')
    ranks=[];negative=[]
    for rank,path in enumerate(config['shards']):
        state=torch.load(path,map_location='cpu',weights_only=True)
        report=compare_states(state,copy.deepcopy(state))
        if report['rank']!=rank:raise ValueError('real shard rank differs')
        ranks.append(report)
        name=next(n for n,p in state['parameters'].items() if p.numel())
        key=state['optimizer']['param_groups'][0]['params'][state['names'].index(name)]
        for field in ('master','exp_avg','exp_avg_sq','step'):
            changed=copy.deepcopy(state)
            t=changed['parameters'][name] if field=='master' else changed['optimizer']['state'][key][field]
            before=t.flatten()[0].clone()
            t.flatten()[0]=torch.nextafter(before,torch.tensor(float('inf')))
            if torch.equal(before,t.flatten()[0]):raise AssertionError('corruption did not change FP32 value')
            bf16_unchanged=torch.equal(before.bfloat16(),t.flatten()[0].bfloat16())
            try:compare_states(state,changed)
            except ValueError as exc:
                if 'tensor differs' not in str(exc):raise
            else:raise AssertionError('corrupted real state accepted')
            negative.append(dict(rank=rank,field=field,rejected=True,bf16_unchanged=bf16_unchanged))
    if not any(x['field']=='master' and x['bf16_unchanged'] for x in negative):
        raise AssertionError('required sub-BF16 negative control absent')
    return dict(passed=True,ranks=ranks,corruptions=negative,
        scope='Real accepted four-rank D step2 layout/self-comparison plus16 mutated-copy rejections. No residency replay, model update or NPU execution.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        atomic_json(a.output/'result.json',run(read(a.config)));atomic_json(a.output/'terminal.json',dict(exit_code=0))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
