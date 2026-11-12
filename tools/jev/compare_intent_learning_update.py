#!/usr/bin/env python3
"""Compare accepted predecessor/update policies on the same learning sources."""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.intent_learning_comparison import compare
from queue_span_native_actor_npu import read,completed
from queue_intent_finite_npu import verify_final,verify_gate
from score_intent_learning_policy import run as score


def run(training,suite):
    training=Path(training);suite=Path(suite);cfg=read(training/'config.json')
    launch=read(training/'launch.json');accepted_launch=read(suite/'launch.json')
    if (accepted_launch['image']!=launch['image'] or not completed(launch['container_id'],launch['image'])
            or not completed(accepted_launch['container_id'],launch['image'])):
        raise ValueError('actual training and independent suite completion required')
    verify_final(suite);verified=read(suite/'verified.json');suite_cfg=read(suite/'config.json')
    if verified['root']!=str(training) or verified['suite']!=str(suite) or suite_cfg['training_root']!=str(training):
        raise ValueError('independent acceptance belongs to another training update')
    if (verified['updates']!=cfg['step']+1 or file_hash(training/'config.json')!=launch['config_sha256']
            or file_hash(suite/'config.json')!=accepted_launch['config_sha256']):
        raise ValueError('accepted update identity differs')
    before=Path(cfg['acceptance_root']);audit=verify_gate(before,cfg['pilot'],cfg['manifest_sha256'])
    reports=[];pins={}
    for root in (before,suite):
        scfg=dict(gate_root=str(root),pilot=cfg['pilot'],manifest_sha256=cfg['manifest_sha256'],
            pins={str(root/p):file_hash(root/p) for p in ('config.json','run/READY.json','run/terminal.json','run/vllm/result.json','run/vllm/terminal.json')})
        reports.append(score(scfg));pins.update(scfg['pins'])
    selection=read(training/'run/selection.json')
    if selection['step']!=cfg['step'] or selection['seed']!=cfg['seed']:
        raise ValueError('selected batch cursor differs')
    if reports[1]['adapter_id']!=verified['adapter_id']:raise ValueError('updated policy differs')
    # Both gate reports independently authenticate the same fixed source release.
    result=compare(*reports,selection['recording_ids'])
    for path in (training/'config.json',training/'run/selection.json',suite/'verified.json',suite/'ACCEPTED.json',Path(__file__),
                 Path(__file__).resolve().parents[2]/'src/multimodal_jev/intent_learning_comparison.py',
                 Path(__file__).resolve().parents[2]/'src/multimodal_jev/intent_policy_metrics.py'):
        pins[str(path)]=file_hash(path)
    return dict(result,training_root=str(training),suite=str(suite),updates=verified['updates'],pins=pins,
        predecessor_gate_audit=audit,full_before=reports[0],full_after=reports[1])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--training',required=True);p.add_argument('--suite',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable comparison already exists')
    atomic_json(a.output,run(a.training,a.suite))
