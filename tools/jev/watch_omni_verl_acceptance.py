#!/usr/bin/env python3
"""Observe one exact CPU comparison and accept its artifacts once, without replay."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from verify_omni_pretrained_verl import accept, read, require, sha


def write(path, value):
    temporary=path.with_suffix('.tmp')
    with temporary.open('w') as handle:
        json.dump(value,handle,indent=2);handle.write('\n');handle.flush();os.fsync(handle.fileno())
    temporary.replace(path)


def inspect(cid):
    require(len(cid)==64 and all(c in '0123456789abcdef' for c in cid), 'exact Docker ID required')
    output=subprocess.check_output(['ssh','-o','ConnectTimeout=8','npu',
        shlex.join(['docker','inspect',cid])],text=True,timeout=30)
    return json.loads(output)[0]


def validate(root, plan_hash):
    require(sha(root/'plan.json')==plan_hash,'observer plan changed')
    plan=read(root/'plan.json');run=Path(plan['run'])
    require(plan['automatic_retry'] is False and plan['timeout_seconds']==15000,
            'bounded no-replay observer required')
    require(sha(run/'launch.json')==plan['launch_sha256']
            and sha(run/'intent.json')==plan['intent_sha256'],'comparison binding changed')
    launch=read(run/'launch.json')
    require(launch['container_id']==plan['container_id'] and launch['image']==plan['image'],
            'exact comparison identity required')
    require(set(plan['source_files'])=={'watch_omni_verl_acceptance.py','verify_omni_pretrained_verl.py'},
            'standalone observer/verifier sources required')
    for name,pin in plan['source_files'].items():
        require(sha(Path(__file__).parent/name)==pin,'observer/verifier source changed')
    return plan


def execute(root, plan_hash):
    plan=validate(root,plan_hash)
    require(not any((root/name).exists() for name in ('verification-intent.json','acceptance.json','result.json')),
            'preserve prior verification; no automatic retry')
    deadline=time.monotonic()+plan['timeout_seconds']
    while True:
        try:state=inspect(plan['container_id'])
        except (subprocess.SubprocessError,json.JSONDecodeError) as exc:
            write(root/'observation.json',dict(error=type(exc).__name__,heartbeat_unix=time.time()))
        else:
            require(state['Id']==plan['container_id'] and state['Image']==plan['image'],'observed identity differs')
            write(root/'observation.json',dict(container_id=state['Id'],state=state['State'],heartbeat_unix=time.time()))
            if state['State']['Status'] in ('exited','dead'):break
        if time.monotonic()>=deadline:
            raise TimeoutError('observation expired; reconcile exact comparison, never replay inference')
        time.sleep(20)
    write(root/'inspection.json',state)
    require(state['State']['Status']=='exited' and state['State']['ExitCode']==0,
            'comparison failed; preserve evidence without replay')
    validate(root,plan_hash)
    with (root/'verification-intent.json').open('x') as handle:
        json.dump(dict(plan_sha256=plan_hash,inspection_sha256=sha(root/'inspection.json'),
                       started_unix=time.time()),handle)
        handle.flush();os.fsync(handle.fileno())
    accepted=accept(Path(plan['run']),plan['launch_sha256'],root/'inspection.json',sha(root/'inspection.json'))
    again=inspect(plan['container_id'])
    require(again['Id']==state['Id'] and again['Image']==state['Image'] and again['State']==state['State'],
            'comparison terminal changed during acceptance')
    write(root/'acceptance.json',accepted)
    write(root/'result.json',dict(container_id=state['Id'],acceptance_sha256=sha(root/'acceptance.json'),
                                weight_updates=0,inference_replayed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--plan-hash',required=True);args=parser.parse_args()
    with (args.root/'observer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not (args.root/'terminal.json').exists(),'preserve prior observer terminal')
        try:execute(args.root,args.plan_hash)
        except BaseException as exc:
            write(args.root/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
        write(args.root/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
