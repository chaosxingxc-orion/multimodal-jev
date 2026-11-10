#!/usr/bin/env python3
"""Original TRAIN audio probe through the frozen transfer inference engine."""
import argparse
import fcntl
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json
from multimodal_jev.omni_slurp_transfer_contract import select_transfer_train
from infer_qwen3_omni_probe import run

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True)
    with (a.output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (a.output/'identity.json').exists():raise FileExistsError('preserve prior attempt')
        try:run(json.loads(a.config.read_text()),a.output,population_selector=select_transfer_train)
        except BaseException as exc:
            atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()))
            raise
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
