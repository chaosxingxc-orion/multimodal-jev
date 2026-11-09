#!/usr/bin/env python3
import argparse,fcntl,json,time
from pathlib import Path
from multimodal_jev.sft import train
from multimodal_jev.baseline import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.output/'result.json').exists():raise FileExistsError('completed run')
        try:train(json.loads(args.config.read_text()),args.output)
        except BaseException as exc:
            atomic_json(args.output/'terminal.json',{'exit_code':1,'error':repr(exc),'finished_unix':time.time()});raise
        atomic_json(args.output/'terminal.json',{'exit_code':0,'finished_unix':time.time()})


if __name__=='__main__':main()
