#!/usr/bin/env python3
"""Detached single-dispatch selected-prompt evaluator using exact baseline ASR evidence."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.frozen_evidence import export_evidence


def observe(cid, image):
    result = subprocess.run(['ssh', '-o', 'ConnectTimeout=8', 'npu',
                             'docker inspect ' + shlex.quote(cid)],
                            capture_output=True, text=True, timeout=20)
    if result.returncode: raise ConnectionError('container observation unavailable')
    item = json.loads(result.stdout)[0]
    if item['Id'] != cid or item['Image'] != image: raise ValueError('container identity changed')
    return item


def wait_container(cid, image, root, phase, deadline):
    while time.time() < deadline:
        try:
            item = observe(cid, image)
        except (ConnectionError, subprocess.TimeoutExpired, json.JSONDecodeError):
            atomic_json(root / 'progress.json', {'phase': phase, 'observation': 'unavailable',
                                                 'heartbeat_unix': time.time()})
            time.sleep(15)
            continue
        atomic_json(root / 'progress.json', {'phase': phase, 'container_id': cid,
                                             'state': item['State'], 'heartbeat_unix': time.time()})
        if item['State']['Status'] in ('exited', 'dead'):
            if item['State']['ExitCode'] != 0 or item['State']['OOMKilled']:
                raise ValueError('dependency or evaluation container failed')
            return item
        time.sleep(15)
    raise TimeoutError('supervisor deadline exceeded; no container restart')


def execute(root):
    plan = json.loads((root / 'plan.json').read_text())
    original = Path(plan['baseline_root'])
    baseline = json.loads((original / 'launch.json').read_text())
    if baseline['container_id'] != plan['baseline_container_id']: raise ValueError('baseline changed')
    selection_path = root / 'selection.json'
    if file_hash(selection_path) != plan['selection_sha256']: raise ValueError('selection changed')
    selection = json.loads(selection_path.read_text())
    if file_hash(root / 'instruction.txt') != selection['selected_instruction_sha256']:
        raise ValueError('selected prompt changed')
    if file_hash(root / 'config-template.json') != plan['config_template_sha256']:
        raise ValueError('registered evaluation configuration changed')
    wait_container(plan['baseline_container_id'], plan['image'], root,
                   'waiting_completed_baseline_for_paired_asr', plan['deadline_unix'])
    export_evidence(original / 'run', root / 'asr-evidence.json')
    config = json.loads((root / 'config-template.json').read_text())
    config['asr_evidence'] = {'path': str(root / 'asr-evidence.json'),
                              'sha256': file_hash(root / 'asr-evidence.json')}
    atomic_json(root / 'config.json', config)
    # The target card must be released by its exact prior experiment.
    prior = observe(plan['prior_device_container_id'], plan['image'])
    if prior['State']['Status'] != 'exited': raise ValueError('target device prior run still active')
    argv = plan['argv']
    launch = {'argv': argv, 'image': plan['image'], 'source_commit': plan['source_commit'],
              'root': str(root), 'physical_device': plan['physical_device'],
              'config_sha256': file_hash(root / 'config.json'),
              'selected_instruction_sha256': selection['selected_instruction_sha256']}
    if (root / 'dispatch-intent.json').exists(): raise ValueError('previous dispatch must be reconciled; no retry')
    atomic_json(root / 'dispatch-intent.json', launch)
    result = subprocess.run(['ssh', '-o', 'ConnectTimeout=8', 'npu', shlex.join(argv)],
                            capture_output=True, text=True, timeout=90)
    (root / 'dispatch.log').write_text(result.stdout + result.stderr)
    if result.returncode: raise RuntimeError('test dispatch failed; preserve intent, no retry')
    cid = result.stdout.strip()
    if len(cid) != 64 or any(c not in '0123456789abcdef' for c in cid): raise ValueError('ambiguous container identity')
    launch['container_id'] = cid
    atomic_json(root / 'launch.json', launch)
    item = wait_container(cid, plan['image'], root, 'selected_prompt_test', plan['deadline_unix'])
    atomic_json(root / 'docker-terminal.json', {'container_id': cid, 'state': item['State']})
    for name, run in [('baseline-score.json', original / 'run'), ('selected-score.json', root / 'run')]:
        with (root / (name + '.log')).open('x') as log:
            subprocess.run([sys.executable, str(root / 'source/tools/jev/score_frozen_requests.py'),
                            '--run', str(run), '--targets', plan['targets'],
                            '--output', str(root / name)] + (['--six-views'] if plan.get('six_views') else []),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    atomic_json(root / 'terminal.json', {'exit_code': 0, 'finished_unix': time.time(),
                                         'scope': 'complete frozen selected prompt and baseline scores; see configuration split/purpose'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    try:
        execute(args.root)
    except BaseException as exc:
        atomic_json(args.root / 'terminal.json', {'exit_code': 1, 'error': repr(exc),
                                                  'finished_unix': time.time(), 'automatic_retry': False})
        raise
