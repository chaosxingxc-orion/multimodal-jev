#!/usr/bin/env python3
"""Wait for exact control containers, validate feedback, call the teacher once."""
import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.teacher_candidates import stage_candidates
from prepare_omni_teacher_feedback import prepare
from configure_omni_teacher import configure, EXECUTOR_SHA
from watch_qwen3_omni_probe import wait


def execute_once(root, executor):
    """Durable local intent precedes process creation; never replay ambiguity."""
    intent = json.loads((root/'teacher-call-intent.json').read_text())
    cfg = root/'executor-config.private.json'
    if file_hash(executor) != EXECUTOR_SHA or file_hash(cfg) != intent['config_sha256']:
        raise ValueError('executor or configured call changed')
    dispatch = root/'process-launch.json'
    if (root/'response').exists():
        raise FileExistsError('prior teacher response state exists')
    with dispatch.open('x') as handle:
        json.dump(dict(dispatch_intent=True, started_unix=time.time(), automatic_retry=False),handle)
    with (root/'process.log').open('xb') as log:
        process = subprocess.Popen([sys.executable,str(executor),'--config',str(cfg)],stdout=log,stderr=log)
        atomic_json(dispatch,dict(dispatch_intent=True,pid=process.pid,started_unix=time.time(),automatic_retry=False))
        code = process.wait()  # Executor owns its finite 900-second total deadline.
    atomic_json(root/'process-terminal.json',dict(exit_code=code,finished_unix=time.time(),automatic_retry=False))
    if code != 0:
        raise RuntimeError('teacher request unsuccessful; inspect retained state, no retry')
    return stage_candidates(root,root/'candidates')


def run(queue):
    plan = json.loads((queue/'queue-intent.json').read_text())
    if (file_hash(plan['executor']) != EXECUTOR_SHA
            or file_hash(plan['reference_config']) != plan['reference_config_sha256']):
        raise ValueError('registered teacher executor or reference changed')
    controls = Path(plan['controls'])
    deadline = time.monotonic()+7500
    for name in ('actual','zeros'):
        launch = json.loads((controls/name/'launch.json').read_text())
        expected = plan['controls_identity'][name]
        if launch['container_id'] != expected['container_id'] or launch['image'] != expected['image']:
            raise ValueError('registered controls changed')
        state = wait(queue,expected['container_id'],expected['image'],deadline)
        if state['Status'] != 'exited' or state['ExitCode'] != 0:
            raise RuntimeError('control container failed; no teacher dispatch')
    # Containers can exit before the owning supervisor writes comparison/terminal.
    terminal = controls/'evaluation-terminal.json'
    finish_deadline = time.monotonic()+180
    while not terminal.exists():
        if time.monotonic()>finish_deadline:
            raise TimeoutError('paired supervisor terminal absent; no teacher dispatch')
        time.sleep(5)
    if json.loads(terminal.read_text())['exit_code'] != 0:
        raise ValueError('paired controls failed')
    if file_hash(plan['reference_config']) != plan['reference_config_sha256']:
        raise ValueError('teacher reference changed while controls were running')
    root = queue/'request'
    prepare(controls,root)
    configure(root,Path(plan['reference_config']),Path(plan['executor']))
    receipt = execute_once(root,Path(plan['executor']))
    atomic_json(queue/'staging-result.json',receipt)
    # Structural staging is not semantic approval; never launch candidates here.


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--queue',type=Path,required=True);a=p.parse_args()
    with (a.queue/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (a.queue/'evaluation-terminal.json').exists() or (a.queue/'request').exists():
            raise FileExistsError('existing attempt; do not replay queue')
        try:
            run(a.queue)
        except BaseException as exc:
            atomic_json(a.queue/'evaluation-terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time(),automatic_retry=False))
            raise
        atomic_json(a.queue/'evaluation-terminal.json',dict(exit_code=0,finished_unix=time.time(),semantic_review='pending'))
