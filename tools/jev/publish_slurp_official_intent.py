#!/usr/bin/env python3
"""Publish full official-clean audio intent data with explicit new admission."""
import argparse
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json
from multimodal_jev.slurp_official_training import publish, read


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('config', 'output', 'state'):
        p.add_argument('--'+name, required=True, type=Path)
    a = p.parse_args()
    a.state.mkdir(exist_ok=False, parents=True)
    try:
        report = publish(read(a.config), a.output,
            lambda v: atomic_json(a.state/'progress.json', dict(**v, updated_unix=time.time())))
        atomic_json(a.state/'result.json', report)
        atomic_json(a.state/'terminal.json', dict(exit_code=0, finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.state/'terminal.json', dict(exit_code=1, error=repr(exc), finished_unix=time.time()))
        raise
