#!/usr/bin/env python3
"""Independent HF/TP1 gate and matched native-actor optimizer audit."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from multimodal_jev.baseline import atomic_json,file_hash


def run(root):
    config=json.loads((root/'config.json').read_text())
    for path,digest in config['pins'].items():
        if file_hash(path)!=digest:raise ValueError('acceptance asset changed: '+path)
    parent=Path(config['native_root'])
    if config['adapter']!=str(parent/'run/adapter'):raise ValueError('native adapter required')
    subprocess.run([sys.executable,str(Path(__file__).with_name('check_span_policy_version.py')),
        '--config',str(root/'config.json'),'--output',str(root/'run')],check=True)
    from verify_span_native_actor_npu import verify_native
    result=verify_native(parent,root,config['reference_root'])
    atomic_json(root/'verified.json',result)
    atomic_json(root/'ACCEPTED.json',dict(passed=True,adapter_id=result['adapter_id'],
        verified_sha256=file_hash(root/'verified.json'),
        maximum_master_and_adam_error=result['maximum_master_and_adam_error'],
        scope=result['scope']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    try:
        run(args.root)
        atomic_json(args.root/'controller-terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(args.root/'controller-terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
