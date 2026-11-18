#!/usr/bin/env python3
"""Restore continuous update3, verify full128, then run scheduled update4."""
import argparse
import json
import os
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json


def run(config, output, rank):
    if (int(os.environ['WORLD_SIZE']) != 4 or int(os.environ['LOCAL_RANK']) != rank
            or config.get('sampled_probe', {}).get('fixture_initialization') != 'restore_continuous_then_one_update'
            or config['sampled_probe'].get('stop_update') != 4
            or config['continuous_resume']['completed_updates'] != 3
            or config['global_batch'] != 128 or config.get('microbatch_per_rank') != 1
            or 'stage2_initialization' in config):
        raise ValueError('explicit engineering DP4 step3 to step4 required')
    from multimodal_jev.local_runtime_storage import require_training_storage
    from probe_simple_continuous_restore import run as restore
    from probe_simple_resident_train import continuous
    require_training_storage(output)
    out = output / f'rank-{rank}'; out.mkdir(parents=True, exist_ok=False)
    try:
        restore(config, output, rank, after_restore=continuous)
        atomic_json(out/'terminal.json', dict(exit_code=0, finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(out/'terminal.json', dict(exit_code=1, error=repr(exc), finished_unix=time.time()))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    run(json.loads(a.config.read_text()), a.output, int(os.environ['RANK']))
