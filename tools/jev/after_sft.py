#!/usr/bin/env python3
"""Host-side dependency runner: Docker terminal state gates evaluation/export.

Run detached on the Docker host. Never treats an old progress file as completion.
Only own named containers are created, and existing ones are observed, not reset.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time


IMAGE='sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'


def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def inspect(name):
    proc=subprocess.run(['docker','inspect',name,'--format','{{json .State}}'],capture_output=True,text=True)
    if proc.returncode:return None
    return json.loads(proc.stdout)


def run(root):
    dependency='multimodal-jev-fsc-audio-sft-init-v1'
    run=Path('/mnt/cephfs/xingchao/experiments/multimodal-jev/fsc-audio-sft-init-v1/run')
    state=inspect(dependency)
    if state is None:raise RuntimeError('SFT dependency container missing; do not restart training')
    while state['Running']:
        save(root/'progress.json',{'phase':'waiting_for_sft','dependency':dependency,
                                  'docker_state':state,'heartbeat_unix':time.time()})
        time.sleep(30)
        state=inspect(dependency)
        if state is None:raise RuntimeError('SFT container disappeared')
    if state['ExitCode']!=0:raise RuntimeError(f'SFT failed: {state}')
    terminal=json.loads((run/'terminal.json').read_text());result=json.loads((run/'result.json').read_text())
    if terminal['exit_code']!=0 or result['status']!='passed' or result['updates']!=47:
        raise RuntimeError('SFT run receipt does not prove completed 47 updates')
    jobs=[('export',[
        'python3','/job/source/tools/jev/export_audio_policy.py','--run',str(run),
        '--output-root','/mnt/cephfs/xingchao/models/multimodal-jev/fsc-sft-init']),
          ('evaluation',[
        'python3','/job/source/tools/jev/evaluate_audio_policy.py',
        '--config','/job/source/configs/experiments/fsc-schema-eval-sft.json','--output','/job/evaluation'])]
    for phase,command in jobs:
        name='multimodal-jev-post-sft-'+phase+'-v1'
        state=inspect(name)
        if state is None:
            argv=['docker','run','-d','--name',name,'--cpus=4','--memory=16g',
                  '-e','PYTHONPATH=/job/source/src',
                  '-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
                  '-v','/mnt/cephfs/xingchao/models:/mnt/cephfs/xingchao/models:'+('rw' if phase=='export' else 'ro'),
                  '-v','/mnt/cephfs/xingchao/datasets:/mnt/cephfs/xingchao/datasets:ro',
                  '-v',str(run)+':'+str(run)+':ro','-v',str(root)+':/job',IMAGE,*command]
            save(root/(phase+'-command.json'),{'argv':argv})
            subprocess.run(argv,check=True)
        while True:
            state=inspect(name)
            if state is None:raise RuntimeError(f'child disappeared: {name}')
            save(root/'progress.json',{'phase':phase,'container':name,'docker_state':state,'heartbeat_unix':time.time()})
            if not state['Running']:break
            time.sleep(30)
        with (root/(phase+'.log')).open('w') as handle:
            subprocess.run(['docker','logs',name],stdout=handle,stderr=subprocess.STDOUT,check=True)
        save(root/(phase+'-terminal.json'),state)
        if state['ExitCode']!=0:raise RuntimeError(f'{phase} failed; see container and retained log')
    save(root/'terminal.json',{'exit_code':0,'finished_unix':time.time()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    root=args.root;root.mkdir(parents=True,exist_ok=True)
    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'terminal.json').exists():raise FileExistsError('terminal supervisor run; inspect before retrying')
        try:
            run(root)
        except BaseException as exc:
            save(root/'terminal.json',{'exit_code':1,'error':repr(exc),'finished_unix':time.time()})
            raise


if __name__=='__main__':main()
