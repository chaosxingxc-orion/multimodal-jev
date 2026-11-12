#!/usr/bin/env python3
"""Read-only supplement for frozen correctness runs whose old gate omitted payload checking."""
import argparse
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.intent_supervision_audit import verify
from multimodal_jev.intent_objective import specification
from multimodal_jev.slurp_intent_pilot import load_pilot
from multimodal_jev.intent_actor_inputs import schedule
from compare_intent_learning_update import run as compare
from queue_span_native_actor_npu import read


def run(training,suite):
    training,suite=Path(training),Path(suite)
    checked=compare(training,suite)  # Actual exits, state and full backend evidence.
    cfg=read(training/'config.json')
    if specification(cfg)!={'name':'correctness-v1'}:
        raise ValueError('supplement only supports original correctness objective')
    population,_=load_pilot(cfg['pilot'],cfg['manifest_sha256'],cfg['arm'])
    selected=schedule(population,cfg['seed'],cfg['step'],cfg['global_requests'])
    if [r['id'] for r in selected]!=read(training/'run/selection.json')['recording_ids']:
        raise ValueError('independent source schedule differs')
    behavior=Path(cfg['acceptance_root'])/'run/vllm/result.json'
    payload=training/'run/supervision.json'
    files=dict(checked['pins'])
    for p in (payload,behavior,Path(cfg['pilot'])/'MANIFEST.json',
              Path(__file__).resolve(),Path(__file__).resolve().parents[2]/'src/multimodal_jev/intent_supervision_audit.py'):
        files[str(p)]=file_hash(p)
    result=verify(selected,read(behavior),read(payload),cfg['arm'])
    for p,h in files.items():
        if file_hash(p)!=h:raise ValueError('evidence changed during audit')
    return dict(result,training_root=str(training),suite_root=str(suite),updates=checked['updates'],files=files)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('training','suite','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable audit already exists')
    atomic_json(a.output,run(a.training,a.suite))
