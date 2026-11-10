#!/usr/bin/env python3
"""One-shot Jev TRAIN probe after the already queued native SFC probe."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

from multimodal_jev.baseline import atomic_json, file_hash
from infer_sfc_jev_omni_probe import select_inputs
from verify_sfc_jev_probe import score
from watch_omni_transfer_readiness import observe_dependency
from watch_sfc_omni_probe import dispatch_once


def check_pins(root, plan):
    for name,value in plan['files'].items():
        if file_hash(root/name)!=value:raise ValueError('prepared Jev artifact changed')
    for name,value in plan['source_files'].items():
        if file_hash(root/'source'/name)!=value:raise ValueError('frozen Jev source changed')
    prep=json.loads((root/'preparation.json').read_text())
    if file_hash(prep['official_scorer'])!=prep['official_scorer_sha256']:
        raise ValueError('official scorer changed')


def execute(root):
    if any((root/name).exists() for name in ('evaluation-terminal.json','dispatch-intent.json','dispatch.log','launch.json','run')):
        raise FileExistsError('prior attempt; no replay')
    plan=json.loads((root/'queue-plan.json').read_text());check_pins(root,plan)
    if plan['dependency_kind']!='native-sfc-train6' or plan['automatic_retry'] is not False:
        raise ValueError('registered native SFC predecessor required')
    deadline=time.monotonic()+100000
    while True:
        try:ready=observe_dependency(plan['dependency'])
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError,json.JSONDecodeError) as exc:
            atomic_json(root/'queue-progress.json',dict(phase='observation_retry',error_type=type(exc).__name__,heartbeat_unix=time.time()))
        else:
            atomic_json(root/'queue-progress.json',dict(phase='waiting_native_sfc',ready=ready,heartbeat_unix=time.time()))
            if ready:break
        if time.monotonic()>deadline:raise TimeoutError('dependency observation expired; no dispatch')
        time.sleep(20)
    check_pins(root,plan)
    predecessor=Path(plan['dependency']['root'])
    accepted=json.loads((predecessor/'acceptance.json').read_text())
    if (accepted['requests']!=6 or accepted['source_groups']!=4 or accepted['weight_updates']!=0
            or accepted['result_sha256']!=file_hash(predecessor/'run/result.json')
            or accepted['score_sha256']!=file_hash(predecessor/'score/metrics.json')
            or accepted['container_id']!=json.loads((predecessor/'launch.json').read_text())['container_id']):
        raise ValueError('complete native SFC engineering acceptance required')
    select_inputs(json.loads((root/'config.json').read_text()))
    dispatch_once(root,plan)
    score(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    with (a.root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (a.root/'evaluation-terminal.json').exists() or (a.root/'dispatch-intent.json').exists():
            raise FileExistsError('preserve previous attempt')
        try:execute(a.root)
        except BaseException as exc:
            atomic_json(a.root/'evaluation-terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time(),automatic_retry=False));raise
        atomic_json(a.root/'evaluation-terminal.json',dict(exit_code=0,finished_unix=time.time()))
