#!/usr/bin/env python3
"""Run independent post-resume HF/vLLM suite, then audit actual shard recovery."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from multimodal_jev.baseline import atomic_json,file_hash


def run(config,root):
    for path,digest in config['pins'].items():
        if file_hash(path)!=digest:raise ValueError('acceptance input changed: '+path)
    parent=Path(config['resume_root'])
    if config['adapter']!=str(parent/'run/adapter'):raise ValueError('resume adapter required')
    subprocess.run([sys.executable,str(Path(__file__).with_name('check_span_policy_version.py')),
        '--config',str(root/'config.json'),'--output',str(root/'run')],check=True)
    from verify_span_fsdp_resume_audio import verify_resume
    result=verify_resume(parent,root)
    atomic_json(root/'verified.json',result)
    atomic_json(root/'ACCEPTED.json',dict(passed=True,adapter_id=result['adapter_id'],
        optimizer_updates=2,verified_sha256=file_hash(root/'verified.json'),
        scope='Fixed-topology step-two recovery plus independent full HF/TP1 verification. Not continuous trainer or task gains.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    try:
        run(json.loads((a.root/'config.json').read_text()),a.root)
        atomic_json(a.root/'controller-terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.root/'controller-terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
