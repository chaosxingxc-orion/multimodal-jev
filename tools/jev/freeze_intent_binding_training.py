#!/usr/bin/env python3
"""Freeze behavior evidence only after observing the exact completed input gate.

No training dispatch, device ownership, retries or acceptance of optimizer state.
"""
import argparse
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.container_pins import validate_bind_pins
from multimodal_jev.intent_binding_resume import validate_config
from queue_span_dp4_resume import check_pins
from queue_span_native_actor_npu import read
from verify_intent_binding_training_gate import verify


def freeze(root,policy_id):
    root=Path(root)
    if any((root/p).exists() for p in ('config.json','launch.json','create-attempt.json','run')):
        raise ValueError('binding training already claimed; never overwrite')
    plan=read(root/'prepared.json');cfg=read(root/'config-template.json')
    validate_config(cfg)
    if (cfg['step']!=0 or cfg['arm']!='D' or plan['root']!=str(root)
            or plan['physical_devices']!=[4,5,6,7] or plan['image']!=cfg['image']):
        raise ValueError('fixed D first binding update required')
    check_pins({str(root/'config-template.json'):plan['config_template_sha256'],
        str(root/'source-manifest.json'):plan['source_manifest_sha256']})
    check_pins(cfg['pins']);check_pins({str(root/'source'/p):h for p,h in read(root/'source-manifest.json').items()})
    gate=Path(cfg['acceptance_root'])
    required=[str(gate/p) for p in ('launch.json','run/READY.json','run/terminal.json',
        'run/hf/result.json','run/hf/terminal.json','run/vllm/result.json','run/vllm/terminal.json')]
    if plan['pending_pins']!=required:raise ValueError('exact complete pending gate evidence required')
    for path in required:cfg['pins'][path]=file_hash(path)
    # No final config publication before actual exit and full population check.
    audit=verify(cfg,policy_id)
    if not audit['actual_exit_observed'] or audit['requests']!=712:
        raise ValueError('actual full binding gate required')
    bindings=validate_bind_pins(plan['create_argv'],cfg['pins'])
    atomic_json(root/'input-gate-observed.json',audit)
    atomic_json(root/'bind-pins.json',bindings)
    atomic_json(root/'config.json',cfg)
    return dict(root=str(root),config_sha256=file_hash(root/'config.json'),policy_id=policy_id,
        gate_container_id=audit['gate_container_id'],status='frozen_not_dispatched')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--policy-id',required=True)
    a=p.parse_args();print(freeze(a.root,a.policy_id))
