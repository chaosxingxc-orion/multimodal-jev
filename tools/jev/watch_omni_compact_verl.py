#!/usr/bin/env python3
"""Wait on the exact compact candidate, then invoke one pinned CPU launcher."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.audio2tool_inference import write_once
from multimodal_jev.omni_compact_acceptance import CONTAINER, IMAGE


def inspect_candidate():
    return json.loads(subprocess.check_output(
        ['ssh', 'npu', 'docker inspect ' + CONTAINER], timeout=45))[0]


def candidate_ready(item):
    if item['Id'] != CONTAINER or item['Image'] != IMAGE:
        raise ValueError('original exact candidate container required')
    state = item['State']
    if state['OOMKilled'] or state['Status'] == 'dead':
        raise RuntimeError('candidate failed; no actor dispatch')
    if state['Status'] == 'exited':
        if state['ExitCode'] != 0: raise RuntimeError('candidate exited unsuccessfully')
        return True
    if state['Status'] not in ('running', 'created', 'paused', 'restarting'):
        raise ValueError('unknown candidate state; inspect manually')
    return False


def check_plan(root, pin):
    if file_hash(root/'plan.json') != pin: raise ValueError('registered queue plan changed')
    plan = json.loads((root/'plan.json').read_text())
    if (plan['schema'] != 'omni-compact-verl-dispatch-v1' or plan['candidate_id'] != CONTAINER
            or plan['image'] != IMAGE or plan['automatic_retry'] is not False
            or plan['dependency_seconds'] != 14400 or plan['launcher_seconds'] != 900
            or plan['python'] != '/usr/bin/python3'
            or plan['cwd'] != '/home/xingchao/workspace/multimodal-jev'):
        raise ValueError('fixed one-shot CPU dispatch contract required')
    if file_hash(root/'launcher.py') != plan['launcher_sha256']:
        raise ValueError('pinned launcher changed')
    for name, sha in plan['source_files'].items():
        if file_hash(root/'source'/name) != sha: raise ValueError('coordinator source changed')
    return plan


def execute(root, pin):
    if any((root/name).exists() for name in ('dispatch-intent.json', 'terminal.json')):
        raise FileExistsError('prior dispatch attempt; never replay')
    plan = check_plan(root, pin)
    deadline = time.monotonic() + plan['dependency_seconds']
    while True:
        try:
            item = inspect_candidate()
            ready = candidate_ready(item)
            atomic_json(root/'progress.json', dict(phase='waiting_candidate', ready=ready,
                container_id=CONTAINER, state=item['State'], heartbeat_unix=time.time()))
            if ready: break
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            atomic_json(root/'progress.json', dict(phase='observation_retry', container_id=CONTAINER,
                error_type=type(exc).__name__, heartbeat_unix=time.time()))
        if time.monotonic() >= deadline:
            raise TimeoutError('candidate observation expired; inspect same container, never replay')
        time.sleep(20)
    plan = check_plan(root, pin)
    write_once(root/'candidate-inspection.json', item)
    # The frozen launcher independently re-inspects and recomputes all six
    # candidate logit comparisons before creating any actor experiment files.
    command = [plan['python'], '-B', str(root/'launcher.py')]
    write_once(root/'dispatch-intent.json', dict(argv=command, created_unix=time.time(), automatic_retry=False))
    atomic_json(root/'progress.json', dict(phase='validating_and_dispatching', heartbeat_unix=time.time()))
    env = dict(os.environ, PYTHONPATH=str(root/'source/src'), TORCH_DEVICE_BACKEND_AUTOLOAD='0',
               OMP_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    with (root/'launcher.log').open('xb') as log:
        result = subprocess.run(command, cwd=plan['cwd'], env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=plan['launcher_seconds'])
    write_once(root/'dispatch-result.json', dict(exit_code=result.returncode, finished_unix=time.time()))
    if result.returncode: raise RuntimeError('launcher failed or ambiguous; inspect artifacts, no replay')
    atomic_json(root/'progress.json', dict(phase='actor_dispatched', heartbeat_unix=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True); p.add_argument('--plan-hash', required=True)
    a = p.parse_args()
    with (a.root/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (a.root/'terminal.json').exists(): raise FileExistsError('preserve prior terminal')
        try: execute(a.root, a.plan_hash)
        except BaseException as exc:
            write_once(a.root/'terminal.json', dict(exit_code=1, error=repr(exc), finished_unix=time.time(), automatic_retry=False))
            raise
        write_once(a.root/'terminal.json', dict(exit_code=0, finished_unix=time.time(), automatic_retry=False))
