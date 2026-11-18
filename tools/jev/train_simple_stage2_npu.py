#!/usr/bin/env python3
"""Formal full-data Stage2 entry over the common actor and resident loop."""
import argparse
import json
import os
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json


def run(config, output, rank):
    from multimodal_jev.simple_stage2_release import authenticate
    from multimodal_jev.local_runtime_storage import require_training_storage
    # Admission precedes imports that construct a model or connect to devices.
    authenticate(config)
    if (int(os.environ['WORLD_SIZE']) != 4 or int(os.environ['LOCAL_RANK']) != rank
            or rank not in range(4) or 'sampled_probe' in config):
        raise ValueError('explicit formal DP4 Stage2 required')
    require_training_storage(output)
    out = output / f'rank-{rank}'; out.mkdir(parents=True, exist_ok=False)
    from probe_simple_sampled_npu import initialize_stage2_actor, resume_stage2_actor
    from probe_simple_resident_train import continuous
    try:
        if 'continuous_resume' in config:
            resume_stage2_actor(config, output, rank, after_resume=continuous)
        else:
            initialize_stage2_actor(config, output, rank, after_initialize=continuous)
        atomic_json(out / 'terminal.json', dict(exit_code=0, finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(out / 'terminal.json', dict(exit_code=1, error=repr(exc), finished_unix=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output, int(os.environ['RANK']))
