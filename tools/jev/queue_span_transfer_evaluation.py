#!/usr/bin/env python3
"""Wait for one existing training container, then audit and launch evaluation."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash


def run(config,output):
    name=config['container']
    # Waiting on this exact live/exited container never restarts training.
    state=json.loads(subprocess.check_output(['ssh','npu','docker','inspect',name],text=True))[0]
    if state['Id']!=config['container_id']:raise ValueError('training container replaced')
    atomic_json(output/'progress.json',dict(stage='waiting_existing_training',container=name,updated_unix=time.time()))
    result=subprocess.run(['ssh','npu','docker','wait',name],capture_output=True,text=True)
    atomic_json(output/'docker-wait.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    if result.returncode!=0 or result.stdout.strip()!='0':raise ValueError('training wait failed or container exited nonzero')
    for path,digest in config['pins'].items():
        if file_hash(path)!=digest:raise ValueError('queued evaluation code changed; requires new reviewed queue')
    atomic_json(output/'progress.json',dict(stage='verifying_and_launching_evaluation',updated_unix=time.time()))
    with (output/'launcher.log').open('x') as log:
        subprocess.run([sys.executable,config['launcher'],'--training',config['training'],
            '--root',config['evaluation'],'--device',str(config['device']),'--launch'],
            stdout=log,stderr=subprocess.STDOUT,check=True,cwd=config['repository'])
    launch=json.loads((Path(config['evaluation'])/'launch.json').read_text())
    if launch['launch_exit_code']!=0:raise ValueError('evaluation launch failed')
    atomic_json(output/'dispatched.json',dict(evaluation=config['evaluation'],container_id=launch['container_id'],
        launch_sha256=file_hash(Path(config['evaluation'])/'launch.json'),scope='Evaluation dispatched; completion and metric verification still required'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(json.loads(a.config.read_text()),a.output)
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
