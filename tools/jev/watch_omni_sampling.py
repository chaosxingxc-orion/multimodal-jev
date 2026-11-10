#!/usr/bin/env python3
"""One-shot frozen sampling after full native SFC TEST, then CPU acceptance."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from multimodal_jev.baseline import atomic_json, file_hash
from watch_omni_transfer_readiness import observe_dependency
from watch_qwen3_omni_probe import inspect, wait
from verify_sfc_omni_full import accept as accept_native


def read(path):return json.loads(Path(path).read_text())


def check_pins(root, plan, expected_hash):
    if file_hash(root/'queue-plan.json')!=expected_hash:
        raise ValueError('registered sampling queue changed')
    if (plan['schema']!='omni-frozen-sampling-queue-v1' or plan['automatic_retry'] is not False
            or plan['physical_devices']!=[6,7] or plan['wall_seconds']!=14400
            or plan['dependency_seconds']!=350000):
        raise ValueError('registered complete sampling queue required')
    for name,pin in plan['files'].items():
        if file_hash(root/name)!=pin:raise ValueError('sampling input changed: '+name)
    prep=read(root/'preparation.json')
    for name,pin in prep['source_files'].items():
        if file_hash(root/'source'/name)!=pin:raise ValueError('sampling source changed')
    if prep['image']!=plan['image'] or prep['source_commit']!=plan['source_commit']:
        raise ValueError('sampling image/source differs')
    for name,argv,expected in (('sampling',prep['argv'],{'/dev/davinci6','/dev/davinci7'}),
                                ('acceptance',plan['acceptance_argv'],set())):
        devices={argv[i+1].split(':')[0] for i,v in enumerate(argv[:-1]) if v=='--device'}
        if ({d for d in devices if d.removeprefix('/dev/davinci').isdigit()}!=expected
                or any(v.startswith('--privileged') or v.startswith('--device=') for v in argv)
                or plan['image'] not in argv or argv[:3]!=['docker','run','-d']):
            raise ValueError(name+' command device/image contract differs')
    pre=Path(plan['preflight_root'])
    if file_hash(pre/'run/result.json')!=plan['preflight_result_sha256']:
        raise ValueError('successful CPU preflight changed')
    result=read(pre/'run/result.json');identity=read(pre/'run/identity.json')
    if (result['status']!='passed' or result['decisions']!=78 or result['model_loaded'] is not False
            or result['identity_sha256']!=file_hash(pre/'run/identity.json')
            or identity['runner_sha256']!=prep['source_files']['tools/jev/sample_sfc_omni_learning.py']):
        raise ValueError('matching full input preflight required')
    for name,pin in identity['source_files'].items():
        if prep['source_files'].get(name)!=pin:raise ValueError('preflight module changed')
    return prep


def require_free_devices():
    for cid in subprocess.check_output(['ssh','npu','docker ps -q'],text=True,timeout=20).split():
        item=inspect(cid)
        if item['HostConfig'].get('Privileged') or any(d['PathOnHost'] in ('/dev/davinci6','/dev/davinci7')
                for d in item['HostConfig'].get('Devices') or []):
            raise RuntimeError('NPU6/7 may be occupied; sampling remains undispatched')


def dispatch(root, name, argv, image, source_commit, seconds, devices):
    intent=root/(name+'-intent.json');launch=root/(name+'-launch.json')
    if intent.exists() or launch.exists():raise FileExistsError('prior dispatch; no replay')
    with intent.open('x') as f:
        json.dump(dict(argv=argv,started_unix=time.time(),automatic_retry=False),f)
        f.flush();os.fsync(f.fileno())
    proc=subprocess.run(['ssh','npu',shlex.join(argv)],capture_output=True,text=True,timeout=90)
    (root/(name+'-dispatch.log')).write_text(proc.stdout+proc.stderr)
    cid=proc.stdout.strip()
    if proc.returncode or len(cid)!=64 or any(c not in '0123456789abcdef' for c in cid):
        raise RuntimeError('ambiguous dispatch; inspect retained intent, no replay')
    record=dict(container_id=cid,image=image,source_commit=source_commit,physical_devices=devices)
    atomic_json(launch,record)
    if name=='sampling':atomic_json(root/'launch.json',record)
    state=wait(root,cid,image,time.monotonic()+seconds)
    atomic_json(root/(name+'-docker-terminal.json'),dict(container_id=cid,state=state))
    logs=subprocess.run(['ssh','npu','docker','logs',cid],capture_output=True,text=True,timeout=30)
    (root/(name+'-container.log')).write_text(logs.stdout+logs.stderr)
    if state['Status']!='exited' or state['ExitCode']!=0:
        raise RuntimeError(name+' container failed; preserve original attempt')
    return cid


def execute(root, expected_hash):
    if any((root/n).exists() for n in ('sampling-intent.json','run','launch.json','evaluation-terminal.json')):
        raise FileExistsError('prior sampling attempt; no replay')
    plan=read(root/'queue-plan.json');check_pins(root,plan,expected_hash)
    deadline=time.monotonic()+plan['dependency_seconds']
    while True:
        try:ready=observe_dependency(plan['dependency'])
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError,json.JSONDecodeError) as exc:
            atomic_json(root/'queue-progress.json',dict(phase='observation_retry',error_type=type(exc).__name__,heartbeat_unix=time.time()))
        else:
            atomic_json(root/'queue-progress.json',dict(phase='waiting_native_full_test',ready=ready,heartbeat_unix=time.time()))
            if ready:break
        if time.monotonic()>deadline:raise TimeoutError('dependency observation expired; no replay')
        time.sleep(20)
    # Read-only acceptance checks, no target labels or score-based scheduling.
    for split in ('test_id','test_ood'):
        accept_native(Path(plan['dependency']['root'])/split)
    prep=check_pins(root,plan,expected_hash)
    with Path(plan['resource_lock']).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require_free_devices()
        cid=dispatch(root,'sampling',prep['argv'],plan['image'],plan['source_commit'],14700,[6,7])
    snapshot=inspect(cid)
    if snapshot['Id']!=cid or snapshot['Image']!=plan['image'] or snapshot['State']['ExitCode']!=0:
        raise ValueError('sampling container identity changed')
    atomic_json(root/'sampling-inspection.json',snapshot)
    check_pins(root,plan,expected_hash)
    argv=plan['acceptance_argv']+['--inspection',str(root/'sampling-inspection.json'),
        '--inspection-hash',file_hash(root/'sampling-inspection.json')]
    validator=dispatch(root,'acceptance',argv,plan['image'],plan['source_commit'],900,[])
    accepted=read(root/'acceptance.json')
    if (accepted['container_id']!=cid or accepted['responses']!=156 or accepted['weight_updates']!=0
            or accepted['result_sha256']!=file_hash(root/'run/result.json')):
        raise ValueError('complete independent sampling acceptance required')
    # Reinspect immutable terminal of the original container after CPU validation.
    after=inspect(cid)
    if after['Id']!=cid or after['Image']!=plan['image'] or after['State']['Status']!='exited' or after['State']['ExitCode']!=0:
        raise ValueError('original sampling container no longer matches')
    atomic_json(root/'queue-progress.json',dict(phase='accepted',responses=156,container_id=cid,
        acceptance_container_id=validator,heartbeat_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--plan-hash',required=True);a=p.parse_args()
    with (a.root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (a.root/'evaluation-terminal.json').exists():raise FileExistsError('preserve prior terminal')
        try:execute(a.root,a.plan_hash)
        except BaseException as exc:
            atomic_json(a.root/'evaluation-terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time(),automatic_retry=False));raise
        atomic_json(a.root/'evaluation-terminal.json',dict(exit_code=0,finished_unix=time.time(),automatic_retry=False))
