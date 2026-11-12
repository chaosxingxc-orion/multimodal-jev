#!/usr/bin/env python3
"""Freeze an already prepared independent suite after exact training exit zero.

    Does not dispatch containers or replace a prior attempt.
"""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.container_pins import validate_bind_pins
from queue_span_dp4_resume import inspect,check_pins


def freeze(root):
    root=Path(root);read=lambda p:json.loads(p.read_text())
    if any((root/p).exists() for p in ('config.json','launch.json','create-attempt.json','run')):
        raise ValueError('suite already claimed; no overwrite')
    plan=read(root/'prepared.json');config=read(root/'config-template.json')
    check_pins({str(root/'config-template.json'):plan['config_template_sha256'],
        str(root/'source-manifest.json'):plan['source_manifest_sha256']})
    check_pins(config['pins'])
    training=Path(config['training_root']);launch=read(training/'launch.json')
    if launch['image']!=plan['image'] or launch['physical_devices']!=[4,5,6,7]:raise ValueError('training identity differs')
    row=inspect([launch['container_id']])[launch['container_id']]
    if row['Image']!=plan['image'] or row['State']['Status']!='exited' or row['State']['ExitCode']!=0:
        raise ValueError('exact training container must exit zero')
    check_pins({str(training/'config.json'):launch['config_sha256'],
        str(training/'source-manifest.json'):launch['source_manifest_sha256']})
    parent=read(training/'config.json');check_pins(parent['pins'])
    if (parent['model_path']!=config['model'] or parent['runtime_profile']!=config['runtime_profile']
            or type(parent['step'])!=int or parent['step']<0):raise ValueError('training contract differs')
    config['pins'].update(parent['pins'])
    for base in (training,root):
        source_pins={str(base/'source'/p):h for p,h in read(base/'source-manifest.json').items()}
        check_pins(source_pins);config['pins'].update(source_pins)
    adapter_id,files=checkpoint_identity(training/'run/adapter')
    config['adapter_id']=adapter_id
    config['pins'].update({str(training/'run/adapter'/p):h for p,h in files.items()})
    config['base_id']=checkpoint_identity(config['model'])[0]
    for rank in range(4):
        out=training/'run'/f'rank-{rank}';result=read(out/'result.json')
        if (read(out/'terminal.json')['exit_code']!=0 or result['passed'] is not True
                or result['rank']!=rank or result['world_size']!=4 or result['updates']!=parent['step']+1
                or result['adapter_id']!=adapter_id or result['arm']!=parent['arm']):
            raise ValueError('successful matching finite update required')
        for name in ('result.json','terminal.json','selection.json','before.json','after.json','optimizer-local.pt'):
            config['pins'][str(out/name)]=file_hash(out/name)
        if parent['step']>0:
            for name in ('restored-local.pt','optimizer-resume.json'):
                config['pins'][str(out/name)]=file_hash(out/name)
    if parent['step']>0:config['pins'][str(training/'run/resume-lineage.json')]=file_hash(training/'run/resume-lineage.json')
    for path in (training/'launch.json',training/'config.json',training/'source-manifest.json',
                 training/'run/selection.json',training/'run/supervision.json'):
        config['pins'][str(path)]=file_hash(path)
    audit=validate_bind_pins(plan['create_argv'],config['pins'])
    atomic_json(root/'bind-pins.json',audit)
    atomic_json(root/'training-terminal-observed.json',dict(container_id=row['Id'],image=row['Image'],state=row['State']))
    atomic_json(root/'config.json',config)
    return dict(root=str(root),config_sha256=file_hash(root/'config.json'),adapter_id=adapter_id,
        pinned_files=len(config['pins']),status='frozen_not_dispatched')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    print(json.dumps(freeze(a.root),indent=2))
