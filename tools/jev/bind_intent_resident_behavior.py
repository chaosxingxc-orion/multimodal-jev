#!/usr/bin/env python3
"""Bind an accepted resident policy to actor inputs, preserving state evidence.

This is an input-consumption overlay. It does not rewrite previous_suite or grant
optimizer restoration without authenticating the original state acceptance.
Online dispatch and in-memory session integration remain separate.
"""
import argparse
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.intent_resident_behavior import authenticate
from queue_span_native_actor_npu import read
from queue_span_dp4_resume import check_pins
from verify_span_resident_policy_suite import verify
from multimodal_jev.intent_resume import authenticate as authenticate_resume


def bind(training_config,audit_path,key):
    training_config,audit_path=map(Path,(training_config,audit_path));original=read(training_config);check_pins(original['pins'])
    if 'resident_behavior' in original:raise ValueError('already bound; do not silently replace behavior')
    continuation='previous_root' in original or 'previous_suite' in original
    if continuation:authenticate_resume(original)
    saved=read(audit_path)
    # Re-observe actual container exit and reconstruct the complete audit. The
    # saved audit is not accepted merely because its own passed field is true.
    fresh=verify(saved['root'],original['pilot'],original['manifest_sha256'],saved['image'])
    if fresh!=saved:raise ValueError('resident audit does not match independent host verification')
    entries=[r for r in fresh['policies'] if r['key']==key]
    if len(entries)!=1:raise ValueError('exact resident policy key required')
    pid=checkpoint_identity(original['model_path'])[0]+':'+checkpoint_identity(original['initial_adapter'])[0]
    if entries[0]['policy_id']!=pid:raise ValueError('actor adapter differs from resident policy')
    cfg=dict(original,pins=dict(original['pins']),acceptance_root=entries[0]['gate_root'],
        resident_behavior=dict(schema='jev-resident-behavior-v1',audit=str(audit_path),key=key))
    cfg['pins'].update(fresh['files']);cfg['pins'][str(audit_path)]=file_hash(audit_path)
    cfg['pins'][str(training_config)]=file_hash(training_config)
    evidence=authenticate(cfg,pid)
    if continuation:
        evidence['resume_state']=authenticate_resume(cfg)
    return cfg,evidence


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('training-config','audit','key','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError('immutable binding exists')
    cfg,evidence=bind(a.training_config,a.audit,a.key);atomic_json(out,cfg)
    atomic_json(out.with_suffix('.binding.json'),evidence)
