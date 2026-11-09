#!/usr/bin/env python3
"""Update one on-policy actor iteration from a completed vLLM rollout run."""
import argparse
import fcntl
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json
from multimodal_jev.rl_run import train


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.output/'result.json').exists():raise FileExistsError('completed actor iteration')
        try:
            atomic_json(args.output/'progress.json',{'phase':'validating_and_updating','heartbeat_unix':time.time()})
            train(json.loads(args.config.read_text()),args.output)
        except BaseException as exc:
            atomic_json(args.output/'terminal.json',{'exit_code':1,'error':repr(exc),'finished_unix':time.time()})
            raise
        atomic_json(args.output/'terminal.json',{'exit_code':0,'finished_unix':time.time()})
