#!/usr/bin/env python3
"""Export full SLURP typed bundles, detached-safe with immutable result."""
import argparse
import fcntl
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json
from multimodal_jev.slurp_jev_full import build

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for key in ('native','destination','state'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();args.state.mkdir(parents=True,exist_ok=True)
    with (args.state/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.state/'terminal.json').exists():raise FileExistsError('previous attempt retained')
        try:
            path,receipt=build(args.native,args.destination,args.state)
            atomic_json(args.state/'result.json',{'path':str(path),**receipt})
        except BaseException as exc:
            atomic_json(args.state/'terminal.json',{'exit_code':1,'error':repr(exc),'finished_unix':time.time()});raise
        atomic_json(args.state/'terminal.json',{'exit_code':0,'finished_unix':time.time()})
