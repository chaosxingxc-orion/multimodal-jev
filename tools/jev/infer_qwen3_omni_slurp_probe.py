#!/usr/bin/env python3
"""Native SLURP audio-to-composite probe with the original fixed TRAIN groups."""
import argparse
import fcntl
import json
import time
from pathlib import Path

from multimodal_jev.baseline import atomic_json
from multimodal_jev.omni_slurp_contract import select_native_probe
from infer_qwen3_omni_probe import run


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=True)
    with (a.output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (a.output/'identity.json').exists():
            raise FileExistsError('preserve prior attempt; no blind retry')
        try:
            run(json.loads(a.config.read_text()),a.output,population_selector=select_native_probe)
        except BaseException as exc:
            atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()))
            raise
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
