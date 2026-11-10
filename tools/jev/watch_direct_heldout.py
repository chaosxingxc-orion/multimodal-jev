#!/usr/bin/env python3
"""Observe exact detached TEST containers, then score once. Never relaunch."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from multimodal_jev.baseline import atomic_json
from multimodal_jev.consumption import canonical
from score_direct_heldout import score


def watch(root):
    launches=json.loads((root/'launches.json').read_text())
    deadline=time.monotonic()+11400
    while True:
        states=[]
        for launch in launches:
            run=Path(launch['root']);cid=launch['container_id']
            try:
                item=json.loads(subprocess.check_output(['ssh','-o','ConnectTimeout=8','npu',
                    'docker inspect '+cid],text=True,timeout=20))[0]
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError,json.JSONDecodeError) as exc:
                states.append(dict(container_id=cid,observation_error=type(exc).__name__,done=False));continue
            if item['Id']!=cid or item['Image']!=launch['image']:raise ValueError('container identity mismatch')
            state=item['State'];done=state['Status'] in ('exited','dead')
            states.append(dict(container_id=cid,state=state,done=done))
            if done:
                atomic_json(run/'docker-terminal.json',dict(container_id=cid,state=state,observed_unix=time.time()))
                if state['ExitCode']!=0:raise RuntimeError('TEST container failed: '+cid)
                terminal=json.loads((run/'run/terminal.json').read_text())
                if terminal['exit_code']!=0:raise RuntimeError('TEST application failed: '+cid)
        atomic_json(root/'observation.json',dict(states=states,heartbeat_unix=time.time()))
        if all(s['done'] for s in states):
            report=score(root)
            with (root/'result.json').open('xb') as f:f.write(canonical(report))
            return
        if time.monotonic()>deadline:raise TimeoutError('observation deadline; no restart')
        time.sleep(20)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    try:watch(a.root)
    except BaseException as exc:
        atomic_json(a.root/'evaluation-terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time(),automatic_retry=False));raise
    atomic_json(a.root/'evaluation-terminal.json',dict(exit_code=0,finished_unix=time.time()))
