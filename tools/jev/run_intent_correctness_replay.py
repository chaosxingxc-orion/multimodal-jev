#!/usr/bin/env python3
"""Fixed-image CPU comparison of real accepted correctness reference states."""
import argparse
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json
from queue_span_native_actor_npu import read
from queue_span_dp4_resume import check_pins


def run(config,out):
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('no-device verifier required')
    check_pins(config['pins'])
    from verify_intent_correctness_replay_cpu import verify
    result=verify(config['reference'],config['reference_suite'],config['replay'],config['replay_suite'])
    atomic_json(out/'verified.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(read(a.config),a.output);atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
