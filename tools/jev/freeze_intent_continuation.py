#!/usr/bin/env python3
"""Bind next-step configuration only to actual accepted predecessor state."""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.container_pins import validate_bind_pins
from multimodal_jev.intent_resume import authenticate
from queue_span_dp4_resume import inspect,check_pins


def freeze(root):
    root=Path(root);read=lambda p:json.loads(p.read_text())
    if any((root/p).exists() for p in ('config.json','create-attempt.json','launch.json','run')):
        raise ValueError('continuation already claimed')
    plan=read(root/'prepared.json');cfg=read(root/'config-template.json')
    check_pins({str(root/'config-template.json'):plan['config_template_sha256'],
        str(root/'source-manifest.json'):plan['source_manifest_sha256']})
    check_pins(cfg['pins'])
    for base,devices in ((Path(cfg['previous_root']),[4,5,6,7]),(Path(cfg['previous_suite']),[4])):
        launch=read(base/'launch.json');row=inspect([launch['container_id']])[launch['container_id']]
        if (launch['image']!=plan['image'] or launch['physical_devices']!=devices
                or row['Image']!=plan['image'] or row['State']['Status']!='exited' or row['State']['ExitCode']!=0):
            raise ValueError('exact predecessor training and acceptance must exit zero')
        check_pins({str(base/'config.json'):launch['config_sha256'],
            str(base/'source-manifest.json'):launch['source_manifest_sha256']})
        parent=read(base/'config.json');check_pins(parent['pins']);cfg['pins'].update(parent['pins'])
        source={str(base/'source'/p):h for p,h in read(base/'source-manifest.json').items()}
        check_pins(source);cfg['pins'].update(source)
        for name in ('launch.json','config.json','source-manifest.json'):cfg['pins'][str(base/name)]=file_hash(base/name)
    suite=Path(cfg['previous_suite']);verified=read(suite/'verified.json')
    check_pins(verified['files']);cfg['pins'].update(verified['files'])
    for name in ('ACCEPTED.json','verified.json','controller-terminal.json','run/READY.json','run/terminal.json',
                 'run/hf/result.json','run/hf/terminal.json','run/vllm/result.json','run/vllm/terminal.json'):
        cfg['pins'][str(suite/name)]=file_hash(suite/name)
    adapter=Path(cfg['initial_adapter']);_,files=checkpoint_identity(adapter)
    cfg['pins'].update({str(adapter/p):h for p,h in files.items()})
    source={str(root/'source'/p):h for p,h in read(root/'source-manifest.json').items()}
    check_pins(source);cfg['pins'].update(source)
    lineage=authenticate(cfg)
    atomic_json(root/'bind-pins.json',validate_bind_pins(plan['create_argv'],cfg['pins']))
    atomic_json(root/'predecessor-verified.json',lineage)
    atomic_json(root/'config.json',cfg)
    return dict(root=str(root),step=cfg['step'],config_sha256=file_hash(root/'config.json'),status='frozen_not_dispatched')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    print(json.dumps(freeze(a.root),indent=2))
