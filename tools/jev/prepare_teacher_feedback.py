#!/usr/bin/env python3
"""Retired transcript-teacher entry point; original code remains in Git/run snapshots."""
import argparse
import json
from pathlib import Path
from multimodal_jev.end_to_end import require_audio_task_route


def prepare(run, targets_path, output, executor_path, reference_config):
    identity = json.loads((Path(run) / 'identity.json').read_text())
    require_audio_task_route(identity['config'])
    raise ValueError('legacy transcript teacher protocol retired; register direct-audio learning feedback first')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('run', 'targets', 'output', 'executor', 'reference-config'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--dispatch', action='store_true')
    args = p.parse_args()
    prepare(args.run, args.targets, args.output, args.executor, args.reference_config)
