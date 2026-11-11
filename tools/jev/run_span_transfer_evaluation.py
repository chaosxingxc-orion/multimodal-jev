#!/usr/bin/env python3
"""Detached fixed-policy evaluation controller with authenticated stage roster."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash


def run(root):
    plan=json.loads((root/'plan.json').read_text())
    if plan['schema']!='jev-transfer-evaluation-plan-v1':raise ValueError('explicit evaluation plan required')
    for path,digest in plan['pins'].items():
        if file_hash(path)!=digest:raise ValueError('evaluation plan input changed')
    if plan['stages']!=['initial-learning','final-learning','initial-selection','final-selection']:
        raise ValueError('fixed complete stage order required')
    for name in plan['stages']:
        stage=root/name
        atomic_json(root/'progress.json',dict(stage=name,updated_unix=time.time()))
        with (stage/'process.log').open('x') as log:
            subprocess.run([sys.executable,str(Path(__file__).with_name('infer_span_rule_baseline.py')),
                '--config',str(stage/'config.json'),'--output',str(stage/'run')],stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([sys.executable,str(Path(__file__).with_name('verify_span_rule_baseline.py')),
            '--root',str(stage),'--output',str(stage/'verified.json')],check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    # Exclusive execution claim prevents a second controller replaying stages.
    with (a.root/'execution-claimed.json').open('x') as f:json.dump(dict(started_unix=time.time()),f)
    try:
        run(a.root)
        atomic_json(a.root/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.root/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
