#!/usr/bin/env python3
"""One-shot bridge from complete TRAIN selection to the registered full TEST."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from multimodal_jev.baseline import atomic_json, file_hash
from watch_qwen3_omni_probe import inspect


def process_start(pid):
    try:
        # comm can contain whitespace or parentheses; fields after its final ')'.
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:
        return None


def observe(selection, plan):
    for name in ('evaluation-launch.json','queue-intent.json','selection-protocol.json'):
        if file_hash(selection/name)!=plan['inputs'][name]:
            raise ValueError('selection identity changed: '+name)
    terminal=selection/'evaluation-terminal.json'
    if terminal.exists():
        if json.loads(terminal.read_text())['exit_code']!=0:
            raise RuntimeError('selection failed; no TEST dispatch')
        queue=json.loads((selection/'queue-intent.json').read_text())
        for key in queue['order']:
            launch=json.loads((selection/key/'launch.json').read_text())
            item=inspect(launch['container_id'])
            if (item['Id']!=launch['container_id'] or item['Image']!=launch['image']
                    or item['State']['Status']!='exited' or item['State']['ExitCode']!=0):
                raise ValueError('exact selection containers must exit successfully')
        return True
    if process_start(plan['selection_pid'])!=plan['selection_process_start']:
        raise RuntimeError('selection supervisor missing or replaced without terminal; inspect, never restart')
    return False


def execute(root):
    plan=json.loads((root/'intent.json').read_text())
    selection=Path(plan['selection_root']);test=Path(plan['test_root'])
    deadline=time.monotonic()+16000
    while True:
        try:
            ready=observe(selection,plan)
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError,json.JSONDecodeError) as exc:
            atomic_json(root/'progress.json',dict(phase='observation_retry',error_type=type(exc).__name__,heartbeat_unix=time.time()))
        else:
            atomic_json(root/'progress.json',dict(phase='selection_complete' if ready else 'waiting_selection',heartbeat_unix=time.time()))
            if ready:break
        if time.monotonic()>deadline:
            raise TimeoutError('selection observation deadline; no TEST dispatch or automatic restart')
        time.sleep(20)
    source=selection/'source'
    for name,digest in plan['source_files'].items():
        if file_hash(source/name)!=digest:raise ValueError('frozen TEST orchestration changed')
    env={**os.environ,'PYTHONPATH':str(source/'src')}
    protocol=root/'test-protocol.json'
    # Exclusive state precedes any operation that could dispatch a TEST container.
    with (root/'transition-intent.json').open('x') as handle:
        json.dump(dict(started_unix=time.time(),automatic_retry=False),handle)
    if test.exists() or protocol.exists():raise FileExistsError('prior TEST preparation; no replay')
    commands=[
        [sys.executable,str(source/'tools/jev/register_omni_test.py'),'--selection-root',str(selection),'--output',str(protocol)],
        [sys.executable,str(source/'tools/jev/run_omni_test_queue.py'),'--root',str(test),'--prepare','--protocol',str(protocol)],
        [sys.executable,str(source/'tools/jev/run_omni_test_queue.py'),'--root',str(test)],
    ]
    for phase,command in zip(('register','prepare','run'),commands):
        atomic_json(root/'progress.json',dict(phase=phase,heartbeat_unix=time.time()))
        with (root/(phase+'.log')).open('xb') as log:
            child=subprocess.Popen(command,env=env,cwd=source,stdout=log,stderr=log)
            atomic_json(root/(phase+'-launch.json'),dict(pid=child.pid,started_unix=time.time(),automatic_retry=False))
            code=child.wait()
        atomic_json(root/(phase+'-terminal.json'),dict(exit_code=code,finished_unix=time.time()))
        if code:raise RuntimeError('TEST '+phase+' failed; preserve attempt')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    with (a.root/'bridge.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (a.root/'terminal.json').exists() or (a.root/'transition-intent.json').exists():
            raise FileExistsError('prior bridge attempt; no automatic replay')
        try:execute(a.root)
        except BaseException as exc:
            atomic_json(a.root/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time(),automatic_retry=False))
            raise
        atomic_json(a.root/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
