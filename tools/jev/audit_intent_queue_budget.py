#!/usr/bin/env python3
"""Read-only measured timing and workload extrapolation, never a deadline."""
import argparse
from datetime import datetime
from pathlib import Path
import statistics
from multimodal_jev.baseline import atomic_json,file_hash
from queue_span_native_actor_npu import read
from queue_span_dp4_resume import inspect,check_pins


def seconds(state):
    if state['Status']!='exited' or state['ExitCode']!=0:raise ValueError('completed successful timing reference required')
    parse=lambda x:datetime.fromisoformat(x.replace('Z','+00:00'))
    value=(parse(state['FinishedAt'])-parse(state['StartedAt'])).total_seconds()
    if value<=0:raise ValueError('positive actual runtime required')
    return value


def estimate(rows,requests_per_policy,policies):
    if not rows or type(requests_per_policy)!=int or requests_per_policy<1 or type(policies)!=int or policies<1:
        raise ValueError('complete timing/workload inputs required')
    for row in rows:
        if row['requests']!=178 or row['training_seconds']<=0 or row['suite_seconds']<=0:raise ValueError('fixed complete178-request references required')
    rates=[r['suite_seconds']/r['requests'] for r in rows]
    return dict(reference_updates=len(rows),training_seconds_median=statistics.median(r['training_seconds'] for r in rows),
        suite_seconds_median=statistics.median(r['suite_seconds'] for r in rows),
        seconds_per_input_for_both_backends=dict(min=min(rates),median=statistics.median(rates),max=max(rates)),
        target_requests_per_backend=requests_per_policy*policies,target_forward_calls=2*requests_per_policy*policies,
        extrapolated_matrix_hours={k:v*requests_per_policy*policies/3600 for k,v in [('min',min(rates)),('median',statistics.median(rates)),('max',max(rates))]},
        scope='Linear workload extrapolation from complete SLURP suite walltimes, including startup/state audit. FSC audio/context/candidate shapes differ; extrema are not confidence bounds or a completion promise. Queue polling/other experiments excluded.')


def run(controller,builder):
    controller,builder=map(Path,(controller,builder));config=read(controller/'config.json');check_pins(config['pins'])
    build=read(builder);check_pins(build['pins']);processor=read(Path(build['processor_root'])/'run/result.json')
    if not processor['passed'] or processor['recordings']!=3793 or processor['requests']!=22758:raise ValueError('full verified FSC workload required')
    entries=config['arms'][0]['steps'][:5]
    if config['arms'][0]['arm']!='Q' or len(entries)!=5:raise ValueError('fixed Q1-5 references required')
    receipts=[];pins={str(controller/'config.json'):file_hash(controller/'config.json'),str(builder):file_hash(builder)}
    for e in entries:
        pair=[]
        for key in ('training_root','suite_root'):
            root=Path(e[key]);r=read(root/'launch.json');pair.append((root,r))
            check_pins({str(root/'config.json'):r['config_sha256'],str(root/'source-manifest.json'):r['source_manifest_sha256']})
            pins[str(root/'launch.json')]=file_hash(root/'launch.json')
        receipts.append(pair)
    states=inspect([r['container_id'] for pair in receipts for root,r in pair]);rows=[]
    from queue_intent_finite_npu import verify_final
    for index,pair in enumerate(receipts,1):
        measured=[]
        for root,r in pair:
            s=states[r['container_id']]
            if s['Image']!=config['image'] or r['image']!=config['image']:raise ValueError('fixed image required')
            measured.append(seconds(s['State']))
        verify_final(pair[1][0]);v=read(pair[1][0]/'verified.json')
        if v['updates']!=index or v['root']!=str(pair[0][0]) or v['full_gate_requests']!=178:raise ValueError('complete paired timing reference required')
        rows.append(dict(update=index,training_seconds=measured[0],suite_seconds=measured[1],requests=178,
            training_container=pair[0][1]['container_id'],suite_container=pair[1][1]['container_id']))
    result=estimate(rows,processor['requests'],7)
    p=Path(build['processor_root'])/'run/result.json';pins[str(p)]=file_hash(p)
    pins[str(Path(__file__).resolve())]=file_hash(__file__)
    return dict(result,rows=rows,files=pins,scheduling_changed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--controller',required=True);p.add_argument('--builder',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable budget audit exists')
    atomic_json(a.output,run(a.controller,a.builder))
