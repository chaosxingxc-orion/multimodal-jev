#!/usr/bin/env python3
import argparse
import fcntl
import json
from pathlib import Path
import time
from multimodal_jev.slurp import build
from multimodal_jev.baseline import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--state',type=Path,required=True);args=p.parse_args()
    args.state.mkdir(parents=True,exist_ok=True)
    with (args.state/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            output,receipt=build(args.root,args.output_root,json.loads(args.config.read_text()),args.state/'progress.json')
            atomic_json(args.state/'result.json',{'path':str(output),**receipt})
        except BaseException as exc:
            atomic_json(args.state/'terminal.json',{'exit_code':1,'error':repr(exc),'finished_unix':time.time()})
            raise
        atomic_json(args.state/'terminal.json',{'exit_code':0,'finished_unix':time.time()})
        print(json.dumps({'path':str(output),'counts':receipt['counts'],'exclusions':receipt['exclusions']},indent=2))


if __name__=='__main__':main()
