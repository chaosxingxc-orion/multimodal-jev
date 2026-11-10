#!/usr/bin/env python3
"""Wait for exact TRAIN78 acceptance and estimate all populations; never dispatch."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

from multimodal_jev.baseline import atomic_json,file_hash
from estimate_sfc_jev_cost import estimate
from watch_omni_transfer_readiness import observe_dependency


def check(root,plan):
    if (plan['schema']!='sfc-jev-cost-observer-v1' or plan['dispatch'] is not False
            or plan['observation_seconds']!=110000
            or set(plan['populations'])!={'learning','selection','test_id','test_ood'}):
        raise ValueError('complete bounded cost-only observation required')
    for name,value in plan['source_files'].items():
        if file_hash(root/'source'/name)!=value:raise ValueError('observer source changed')
    for spec in plan['populations'].values():
        if file_hash(spec['path'])!=spec['sha256']:raise ValueError('registered population changed')


def execute(root):
    plan=json.loads((root/'observer-plan.json').read_text());check(root,plan)
    if (root/'estimation-intent.json').exists() or (root/'estimates').exists():
        raise FileExistsError('preserve prior cost observation')
    deadline=time.monotonic()+plan['observation_seconds']
    while True:
        try:ready=observe_dependency(plan['dependency'])
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError,json.JSONDecodeError) as exc:
            atomic_json(root/'progress.json',dict(phase='observation_retry',error_type=type(exc).__name__,heartbeat_unix=time.time()))
        else:
            atomic_json(root/'progress.json',dict(phase='waiting_train78',ready=ready,heartbeat_unix=time.time()))
            if ready:break
        if time.monotonic()>deadline:raise TimeoutError('observation expired; no inference restart')
        time.sleep(20)
    check(root,plan)
    with (root/'estimation-intent.json').open('x') as handle:
        json.dump(dict(started_unix=time.time(),dispatch=False),handle)
    out=root/'estimates';out.mkdir(exist_ok=False);reports={}
    for name,spec in plan['populations'].items():
        report=estimate(Path(plan['dependency']['root']),Path(spec['path']),spec['sha256'])
        expected_split='train_id' if name in ('learning','selection') else name
        if (report['split']!=expected_split or (report['partition'] is None)!=(name.startswith('test_'))
                or (report['partition'] is not None and report['partition']['part']!=name)):
            raise ValueError('cost report population differs')
        atomic_json(out/(name+'.json'),report)
        reports[name]={k:report[k] for k in ('population_decisions','shard_count','estimated_serial_wall_days',
            'estimated_npu_device_hours','observed_min_sensitivity_wall_days','observed_max_sensitivity_wall_days')}
        atomic_json(root/'progress.json',dict(phase='estimating',completed=list(reports),heartbeat_unix=time.time()))
    atomic_json(root/'result.json',dict(populations=reports,dispatched=False,weight_updates=0,
        scope='Descriptive parent costs from accepted TRAIN78; not a budget, candidate estimate or completion. '
              'Reports preserve full populations. No inference, teacher or training dispatched.'))
    atomic_json(root/'progress.json',dict(phase='complete_cost_only',heartbeat_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    with (a.root/'observer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (a.root/'terminal.json').exists():raise FileExistsError('existing observer terminal')
        try:execute(a.root)
        except BaseException as exc:
            atomic_json(a.root/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time(),dispatched=False));raise
        atomic_json(a.root/'terminal.json',dict(exit_code=0,finished_unix=time.time(),dispatched=False))
