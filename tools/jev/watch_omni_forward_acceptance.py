#!/usr/bin/env python3
"""Observe one exact forward container, then launch its pinned CPU verifier once."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    temp=path.with_suffix('.tmp')
    with temp.open('w') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(path)


def remote(argv):
    return subprocess.check_output(['ssh','-o','ConnectTimeout=8','npu',shlex.join(argv)],text=True,timeout=90).strip()


def inspect(cid):
    if len(cid)!=64 or any(c not in '0123456789abcdef' for c in cid):raise ValueError('exact Docker ID required')
    return json.loads(remote(['docker','inspect',cid]))[0]


def observe(root,cid,image,seconds):
    deadline=time.monotonic()+seconds
    while True:
        try:state=inspect(cid)
        except (subprocess.SubprocessError,json.JSONDecodeError) as exc:
            write(root/'observation.json',dict(container_id=cid,error=type(exc).__name__,heartbeat_unix=time.time()))
        else:
            if state['Id']!=cid or state['Image']!=image:raise ValueError('container identity differs')
            write(root/'observation.json',dict(container_id=cid,state=state['State'],heartbeat_unix=time.time()))
            if state['State']['Status'] in ('exited','dead'):return state
        if time.monotonic()>deadline:raise TimeoutError('observation expired; reconcile original container, no replay')
        time.sleep(20)


def check(root,pin):
    if sha(root/'plan.json')!=pin:raise ValueError('acceptance plan changed')
    plan=read(root/'plan.json');run=Path(plan['run'])
    if (plan['schema']!='omni-pretrained-forward-offline-acceptance-plan-v1'
            or plan['automatic_retry'] is not False or plan['dispatched'] is not False
            or sha(run/'launch.json')!=plan['launch_sha256'] or sha(run/'intent.json')!=plan['intent_sha256']
            or read(run/'launch.json')['container_id']!=plan['container_id']):
        raise ValueError('original frozen run binding required')
    for name,value in plan['source_files'].items():
        if sha(root/'source'/name)!=value:raise ValueError('verifier source changed')
    argv=plan['argv_prefix']
    if (argv[:2]!=['docker','create'] or plan['image'] not in argv
            or '--network=none' not in argv or '--cpus=2' not in argv or '--memory=8g' not in argv
            or any(v.startswith(('--device','--privileged')) for v in argv)
            or argv[-2:]!=['--inspection',str(root/'inspection.json')]):
        raise ValueError('pinned CPU verifier command required')
    return plan


def execute(root,pin):
    plan=check(root,pin)
    if any((root/n).exists() for n in ('verifier-intent.json','verifier-launch.json','acceptance.json')):
        raise FileExistsError('prior acceptance dispatch; reconcile without replay')
    original=observe(root,plan['container_id'],plan['image'],15000)
    write(root/'original-terminal.json',original)
    if original['State']['Status']!='exited' or original['State']['ExitCode']!=0:
        raise RuntimeError('original forward failed; verifier not dispatched')
    check(root,pin)
    write(root/'inspection.json',original)
    argv=plan['argv_prefix']+['--inspection-hash',sha(root/'inspection.json')]
    with (root/'verifier-intent.json').open('x') as f:
        json.dump(dict(argv=argv,plan_sha256=pin,started_unix=time.time(),automatic_retry=False),f)
        f.flush();os.fsync(f.fileno())
    cid=remote(argv)
    if len(cid)!=64 or any(c not in '0123456789abcdef' for c in cid):raise ValueError('ambiguous create; no replay')
    write(root/'verifier-launch.json',dict(container_id=cid,image=plan['image']))
    # Start is issued once. If SSH disconnects, retain the ID for reconciliation.
    remote(['docker','start',cid])
    terminal=observe(root,cid,plan['image'],900)
    write(root/'verifier-terminal.json',terminal)
    (root/'verifier.log').write_text(remote(['docker','logs',cid])+'\n')
    if terminal['State']['Status']!='exited' or terminal['State']['ExitCode']!=0:
        raise RuntimeError('independent verifier failed; preserve its output')
    accepted=read(root/'acceptance.json')
    if (accepted['container_id']!=plan['container_id'] or accepted['forwards']!=6
            or accepted['weight_updates']!=0 or accepted['inspection_sha256']!=sha(root/'inspection.json')):
        raise ValueError('independent acceptance binding differs')
    again=inspect(plan['container_id'])
    if again['Id']!=original['Id'] or again['Image']!=original['Image'] or again['State']!=original['State']:
        raise ValueError('original terminal changed during verification')
    write(root/'observer-result.json',dict(original_container=plan['container_id'],verifier_container=cid,
        acceptance_sha256=sha(root/'acceptance.json'),forwards=6,weight_updates=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan-hash',required=True);a=p.parse_args()
    with (a.root/'observer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (a.root/'observer-terminal.json').exists():raise FileExistsError('preserve prior observer terminal')
        try:execute(a.root,a.plan_hash)
        except BaseException as exc:
            write(a.root/'observer-terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time(),automatic_retry=False));raise
        write(a.root/'observer-terminal.json',dict(exit_code=0,finished_unix=time.time(),automatic_retry=False))
