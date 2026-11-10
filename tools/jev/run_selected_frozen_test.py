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
from multimodal_jev.end_to_end import require_audio_task_route


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
    require_audio_task_route(json.loads((root / 'config-template.json').read_text()))
    raise ValueError('legacy ASR evidence supervisor retired; use a direct-audio launch')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    try:
        execute(args.root)
    except BaseException as exc:
        atomic_json(args.root / 'terminal.json', {'exit_code': 1, 'error': repr(exc),
                                                  'finished_unix': time.time(), 'automatic_retry': False})
        raise
