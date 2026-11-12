#!/usr/bin/env python3
"""Wait for exact SFT evaluation, check card ownership, create then start probe."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
from multimodal_jev.baseline import atomic_json,file_hash
from prepare_span_pair_launch import prepare

SSH=['ssh','-o','ControlMaster=no','-o','ControlPath=none','npu']
IMAGE='sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'
FD_CHECK='''import os,stat,json
target=os.stat('/host-dev/davinci5').st_rdev
busy=[];denied=[]
for pid in os.listdir('/proc'):
 if not pid.isdigit():continue
 try:
  for fd in os.listdir('/proc/'+pid+'/fd'):
   try:
    s=os.stat('/proc/'+pid+'/fd/'+fd)
    if stat.S_ISCHR(s.st_mode) and s.st_rdev==target:busy.append([int(pid),fd])
   except FileNotFoundError:pass
   except PermissionError:denied.append([pid,fd])
 except FileNotFoundError:pass
 except PermissionError:denied.append([pid,'fd'])
print(json.dumps(dict(busy=busy,denied=denied)))
raise SystemExit(1 if busy or denied else 0)
'''


def check_card5():
    """Reject other Docker reservations and host processes, including idle ones."""
    ids=subprocess.check_output(SSH+['docker','ps','-q'],text=True).split()
    if ids:
        containers=json.loads(subprocess.check_output(SSH+['docker','inspect',*ids],text=True))
        for c in containers:
            host=c['HostConfig']
            if host.get('Privileged') or any(d['PathOnHost']=='/dev/davinci5' for d in host.get('Devices') or []):
                raise ValueError('physical card5 reserved by running container: '+c['Id'])
    # fuser -v may print explanatory output, so use returncode as authoritative.
    # Missing utility or SSH errors must not be mistaken for an empty device.
    r=subprocess.run(SSH+['fuser','/dev/davinci5'],capture_output=True,text=True)
    if r.returncode!=1 or r.stdout.strip() or r.stderr.strip():
        raise ValueError('card5 is busy or ownership check failed: '+repr((r.returncode,r.stdout,r.stderr)))
    # Host SSH user cannot inspect all root-owned descriptors. This short-lived
    # root container only stats host process fds; it never opens the NPU device.
    argv=['docker','run','--rm','--pull=never','--network=none','--pid=host','--user=0',
          '--cap-add=SYS_PTRACE','--security-opt','apparmor=unconfined','--cpus=1','--memory=1g',
          '-v','/dev:/host-dev:ro',IMAGE,'python3','-c',FD_CHECK]
    r=subprocess.run(SSH+[shlex.join(argv)],capture_output=True,text=True)
    if r.returncode!=0:raise ValueError('host descriptor check failed or card busy: '+r.stdout+r.stderr)
    report=json.loads(r.stdout)
    if report!=dict(busy=[],denied=[]):raise ValueError('host descriptors not clear')


def run(config,output):
    def pins():
        for path,digest in config['pins'].items():
            if file_hash(path)!=digest:raise ValueError('queued source or evidence changed')
    pins()
    waiting=config['wait_container_id']
    state=json.loads(subprocess.check_output(SSH+['docker','inspect',waiting],text=True))[0]
    if state['Id']!=waiting:raise ValueError('evaluation container replaced')
    atomic_json(output/'progress.json',dict(stage='waiting_existing_sft_evaluation',container_id=waiting,updated_unix=time.time()))
    r=subprocess.run(SSH+['docker','wait',waiting],capture_output=True,text=True)
    atomic_json(output/'wait.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
    if r.returncode!=0 or r.stdout.strip()!='0':raise ValueError('evaluation wait failed or nonzero exit')
    pins()
    root=Path(config['training_root'])
    # Shared advisory lock serializes this project's competing launches. Docker
    # and fuser checks also reject unrelated work; no process is killed.
    with Path(config['device_lock']).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check_card5()
        preflight=json.loads(Path(config['preflight']).read_text())
        if preflight.get('passed') is not True:raise ValueError('passed CPU preflight required')
        for path,digest in preflight['files'].items():
            if file_hash(path)!=digest:raise ValueError('preflight evidence changed')
        result=json.loads((Path(preflight['root'])/'result.json').read_text())
        plan=prepare(root)
        frozen=json.loads(Path(config['launch_plan']).read_text())
        if plan!=frozen or result['source_manifest_sha256']!=plan['receipt_fields']['source_manifest_sha256']:
            raise ValueError('prepared launch differs from preflight/queued plan')
        atomic_json(output/'progress.json',dict(stage='creating_container',updated_unix=time.time()))
        r=subprocess.run(SSH+[shlex.join(plan['create_argv'])],capture_output=True,text=True)
        atomic_json(output/'create.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
        if r.returncode!=0 or not re.fullmatch('[0-9a-f]{64}',r.stdout.strip()):
            raise ValueError('create failed; inspect exact container name before any retry')
        container=r.stdout.strip()
        receipt={**plan['receipt_fields'],'argv':plan['create_argv'],'container_id':container,
                 'launch_exit_code':0,'stderr':r.stderr,'execution':'docker create then start; see start.json'}
        atomic_json(root/'launch.json',receipt)
        atomic_json(Path(config['launch_receipt']),receipt)
        # Recheck immediately before start; a created container does not yet use
        # the card. A failure leaves its identity available for explicit recovery.
        check_card5()
        r=subprocess.run(SSH+['docker','start',container],capture_output=True,text=True)
        atomic_json(root/'start.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
        if r.returncode!=0:raise ValueError('start failed; do not recreate container')
        actual=json.loads(subprocess.check_output(SSH+['docker','inspect',container],text=True))[0]
        if actual['Id']!=container or actual['Image']!=plan['receipt_fields']['image']:
            raise ValueError('started container identity differs')
        if actual['State']['Status'] not in ('running','exited'):
            raise ValueError('container did not start')
        if actual['State']['Status']=='exited' and actual['State']['ExitCode']!=0:
            raise ValueError('new controller already failed')
        atomic_json(output/'dispatched.json',dict(container_id=container,root=str(root),state=actual['State']['Status'],
            scope='Paired engineering container dispatched; training and post-export gate completion still required.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(json.loads(a.config.read_text()),a.output)
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
