#!/usr/bin/env python3
"""Observe exact gate completion and authenticate all transformed learning views.

This does not dispatch training, accept optimizer state, or supersede an update
verifier. A process observation is kept separate from artifact-only validation.
"""
import argparse
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.intent_binding_gate import authenticate_artifacts
from queue_span_native_actor_npu import completed,read


def verify(config,policy_id):
    path=Path(config['acceptance_root'])/'launch.json'
    if config['pins'].get(str(path))!=file_hash(path):raise ValueError('exact pinned launch required')
    launch=read(path)
    if launch['image']!=config['image'] or not completed(launch['container_id'],config['image']):
        raise ValueError('actual successful gate exit required')
    _,_,audit=authenticate_artifacts(config,policy_id)
    if audit['gate_container_id']!=launch['container_id']:raise ValueError('gate observation identity changed')
    return dict(audit,actual_exit_observed=True,actual_exit_code=0,image=config['image'],
        scope='Full transformed policy input/probability gate, not an optimizer or post-update acceptance.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--policy-id',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable gate observation already exists')
    result=verify(read(a.config),a.policy_id);result['config_sha256']=file_hash(a.config)
    atomic_json(a.output,result)
