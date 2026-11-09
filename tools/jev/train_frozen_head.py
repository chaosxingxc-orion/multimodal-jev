#!/usr/bin/env python3
import argparse
import fcntl
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json, train_head


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.output/'result.json').exists():
            raise FileExistsError('completed run must not be overwritten')
        try:
            receipt=train_head(json.loads(args.config.read_text()),args.output)
        except BaseException as exc:
            atomic_json(args.output/'terminal.json',{'exit_code':1,'error':repr(exc),'finished_unix':time.time()})
            raise
        atomic_json(args.output/'terminal.json',{'exit_code':0,'finished_unix':time.time()})
        print(json.dumps(receipt,indent=2))


if __name__=='__main__': main()
