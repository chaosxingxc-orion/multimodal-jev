#!/usr/bin/env python3
"""Wait for exact evaluation containers and run the full independent comparison."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash


def run(config,output):
    if set(config['evaluations'])!={'sft','rl','conditional'}:
        raise ValueError('exactly three evaluations required')
    ssh=['ssh','-o','ControlMaster=no','-o','ControlPath=none','npu']
    # Bind all identities before blocking on any one container.
    for method,entry in config['evaluations'].items():
        launch=Path(entry['root'])/'launch.json'
        if file_hash(launch)!=entry['launch_sha256']:
            raise ValueError('evaluation launch changed')
        receipt=json.loads(launch.read_text())
        if receipt['container_id']!=entry['container_id'] or receipt['launch_exit_code']!=0:
            raise ValueError('evaluation receipt identity differs')
        state=json.loads(subprocess.check_output(ssh+['docker','inspect',entry['container_id']],text=True))[0]
        if state['Id']!=entry['container_id']:
            raise ValueError('evaluation container replaced')
    for method,entry in config['evaluations'].items():
        atomic_json(output/'progress.json',dict(stage='waiting_evaluation',method=method,
                    container_id=entry['container_id'],updated_unix=time.time()))
        result=subprocess.run(ssh+['docker','wait',entry['container_id']],capture_output=True,text=True)
        atomic_json(output/(method+'-wait.json'),dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode!=0 or result.stdout.strip()!='0':
            raise ValueError('evaluation wait failed or nonzero exit: '+method)
    for path,digest in config['pins'].items():
        if file_hash(path)!=digest:raise ValueError('comparison source changed')
    if Path(config['result']).exists():raise ValueError('exclusive result required')
    atomic_json(output/'progress.json',dict(stage='auditing_and_comparing',updated_unix=time.time()))
    command=[sys.executable,config['comparator']]
    for method,entry in config['evaluations'].items():command+=['--'+method,entry['root']]
    command+=['--output',config['result']]
    with (output/'comparison.log').open('x') as log:
        subprocess.run(command,cwd=config['repository'],stdout=log,stderr=subprocess.STDOUT,check=True)
    report=json.loads(Path(config['result']).read_text())
    if report.get('passed') is not True:raise ValueError('comparison did not pass')
    atomic_json(output/'completed.json',dict(result=config['result'],sha256=file_hash(config['result']),
        scope='Authenticated fixed-policy comparison completed; scientific interpretation still required.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(json.loads(a.config.read_text()),a.output)
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
