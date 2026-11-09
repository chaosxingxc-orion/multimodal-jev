#!/usr/bin/env python3
"""Host-side SLURP evaluation dependency; existing training is only observed."""
import argparse,fcntl,json,subprocess,time
from pathlib import Path
from after_sft import inspect,save,IMAGE


def run(root):
    dependency='multimodal-jev-fsc-audio-sft-init-v1'
    training=Path('/mnt/cephfs/xingchao/experiments/multimodal-jev/fsc-audio-sft-init-v1/run')
    while True:
        state=inspect(dependency)
        if state is None:raise RuntimeError('dependency missing; inspect, never restart automatically')
        save(root/'progress.json',{'phase':'waiting_for_sft','state':state,'heartbeat_unix':time.time()})
        if not state['Running']:break
        time.sleep(30)
    if state['ExitCode']!=0:raise RuntimeError('SFT failed')
    terminal=json.loads((training/'terminal.json').read_text());result=json.loads((training/'result.json').read_text())
    if terminal['exit_code']!=0 or result['status']!='passed' or result['updates']!=47:
        raise RuntimeError('SFT completion not proven')
    name='multimodal-jev-slurp-transfer-eval-sft-v1'
    if inspect(name) is None:
        argv=['docker','run','-d','--name',name,'--cpus=4','--memory=16g','-e','PYTHONPATH=/job/source/src',
              '-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
              '-v','/mnt/cephfs/xingchao/models:/mnt/cephfs/xingchao/models:ro',
              '-v','/mnt/cephfs/xingchao/datasets:/mnt/cephfs/xingchao/datasets:ro',
              '-v',str(training)+':'+str(training)+':ro','-v',str(root)+':/job',IMAGE,
              'python3','/job/source/tools/jev/evaluate_audio_policy.py','--config',
              '/job/source/configs/experiments/slurp-transfer-eval-sft.json','--output','/job/run']
        save(root/'command.json',{'argv':argv});subprocess.run(argv,check=True)
    while True:
        state=inspect(name)
        if state is None:raise RuntimeError('evaluation container disappeared')
        save(root/'progress.json',{'phase':'evaluation','state':state,'heartbeat_unix':time.time()})
        if not state['Running']:break
        time.sleep(30)
    with (root/'container.log').open('w') as log:
        subprocess.run(['docker','logs',name],stdout=log,stderr=subprocess.STDOUT,check=True)
    if state['ExitCode']!=0:raise RuntimeError('evaluation failed')
    if json.loads((root/'run/terminal.json').read_text())['exit_code']!=0:raise RuntimeError('failed evaluation receipt')
    save(root/'terminal.json',{'exit_code':0,'finished_unix':time.time()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=True)
    with (a.root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (a.root/'terminal.json').exists():raise FileExistsError('terminal supervisor')
        try:run(a.root)
        except BaseException as exc:
            save(a.root/'terminal.json',{'exit_code':1,'error':repr(exc),'finished_unix':time.time()});raise
