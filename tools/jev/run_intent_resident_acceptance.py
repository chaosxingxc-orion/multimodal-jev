#!/usr/bin/env python3
"""Run a fresh full gate, then independent resident replay tensor verification."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
from multimodal_jev.baseline import atomic_json,file_hash
from queue_span_native_actor_npu import read
from queue_span_dp4_resume import check_pins


def run(root):
    cfg=read(root/'config.json');check_pins(cfg['pins'])
    subprocess.run([sys.executable,str(Path(__file__).with_name('check_slurp_native_span_policy.py')),
        '--config',str(root/'config.json'),'--output',str(root/'run')],check=True)
    from verify_intent_resident_replay_cpu import verify
    result=verify(cfg['resident_training_root'],root)
    atomic_json(root/'verified.json',result)
    atomic_json(root/'ACCEPTED.json',dict(passed=True,adapter_id=result['adapter_id'],verified_sha256=file_hash(root/'verified.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    try:
        run(a.root);atomic_json(a.root/'controller-terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.root/'controller-terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
