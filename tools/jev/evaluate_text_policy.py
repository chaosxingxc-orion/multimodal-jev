#!/usr/bin/env python3
"""Run a frozen transcript decision control with raw outputs and terminal state."""
import argparse
import fcntl
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json
from multimodal_jev.text_baseline import run


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (a.output/'terminal.json').exists():raise FileExistsError('terminal run; inspect before retry')
        try:run(json.loads(a.config.read_text()),a.output)
        except BaseException as exc:
            atomic_json(a.output/'terminal.json',{'exit_code':1,'error':repr(exc),'finished_unix':time.time()});raise
        atomic_json(a.output/'terminal.json',{'exit_code':0,'finished_unix':time.time()})
