#!/usr/bin/env python3
"""Prepare learning feedback once after exact TRAIN78 acceptance; no API call."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.sfc_teacher_wire import prepared_feedback
from prepare_sfc_jev_teacher_feedback import prepare
from watch_omni_transfer_readiness import observe_dependency


def check(root,plan):
    if (plan['schema']!='sfc-jev-feedback-observer-v1' or plan['API_dispatch'] is not False
            or plan['observation_seconds']!=110000 or plan['automatic_retry'] is not False):
        raise ValueError('bounded feedback-only observer required')
    for name,value in plan['source_files'].items():
        if file_hash(root/'source'/name)!=value:raise ValueError('observer source changed')
    if file_hash(plan['partition'])!=plan['partition_sha256']:
        raise ValueError('registered learning partition changed')


def execute(root):
    plan=json.loads((root/'observer-plan.json').read_text());check(root,plan)
    if (root/'preparation-intent.json').exists() or (root/'feedback').exists():
        raise FileExistsError('preserve prior feedback preparation')
    deadline=time.monotonic()+plan['observation_seconds']
    while True:
        try:ready=observe_dependency(plan['dependency'])
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError,json.JSONDecodeError) as exc:
            atomic_json(root/'progress.json',dict(phase='observation_retry',error_type=type(exc).__name__,heartbeat_unix=time.time()))
        else:
            atomic_json(root/'progress.json',dict(phase='waiting_train78',ready=ready,heartbeat_unix=time.time()))
            if ready:break
        if time.monotonic()>deadline:raise TimeoutError('dependency observation expired; no restart')
        time.sleep(20)
    check(root,plan)
    with (root/'preparation-intent.json').open('x') as handle:
        json.dump(dict(started_unix=time.time(),API_dispatch=False,automatic_retry=False),handle)
        handle.flush();os.fsync(handle.fileno())
    output=root/'feedback'
    receipt=prepare(Path(plan['dependency']['root']),Path(plan['partition']),plan['partition_sha256'],output)
    # Validate the exact downstream wire adapter's contract before declaring ready.
    prepared_feedback(output,file_hash(output/'prepared.json'),plan['partition_sha256'])
    atomic_json(root/'result.json',dict(feedback_root=str(output),prepared_sha256=file_hash(output/'prepared.json'),
        receipt=receipt,API_sent=False,weight_updates=0,
        scope='Exact accepted TRAIN78 learning feedback prepared and checked for teacher configuration. '
              'No executor configuration, API call, candidate acceptance, inference or training.'))
    atomic_json(root/'progress.json',dict(phase='feedback_ready_no_api',heartbeat_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    with (a.root/'observer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (a.root/'terminal.json').exists():raise FileExistsError('existing observer terminal')
        try:execute(a.root)
        except BaseException as exc:
            atomic_json(a.root/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time(),API_sent=False));raise
        atomic_json(a.root/'terminal.json',dict(exit_code=0,finished_unix=time.time(),API_sent=False))
