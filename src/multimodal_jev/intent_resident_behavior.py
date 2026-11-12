"""Consume host-authenticated resident inference without granting state resume.

The host binder recomputes the resident suite audit against actual Docker exit.
This actor-side check authenticates the immutable audit and every recorded file;
it cannot independently query Docker from an isolated training container.
"""
import json
from pathlib import Path
from .baseline import file_hash


def authenticate(config,policy_id):
    binding=config['resident_behavior']
    if (not isinstance(binding,dict) or set(binding)!={'schema','audit','key'}
            or binding['schema']!='jev-resident-behavior-v1'):
        raise ValueError('explicit resident behavior binding required')
    pins=config['pins']
    def pinned(path):
        path=Path(path)
        if pins.get(str(path))!=file_hash(path):raise ValueError('resident behavior evidence missing or changed: '+str(path))
        return json.loads(path.read_text())
    audit=pinned(binding['audit'])
    if (audit.get('passed') is not True or audit['pilot']!=config['pilot']
            or audit['manifest_sha256']!=config['manifest_sha256']):
        raise ValueError('resident audit population differs')
    root=Path(audit['root']);launch=pinned(root/'launch.json');plan=pinned(root/'prepared.json')
    if (launch['container_id']!=audit['container_id'] or launch['image']!=audit['image']
            or launch['image']!=plan['image'] or launch['physical_devices']!=[4]
            or launch['argv']!=plan['create_argv'] or launch['launch_exit_code']!=0):
        raise ValueError('resident audit launch identity differs')
    for path,digest in audit['files'].items():
        if pins.get(path)!=digest or file_hash(path)!=digest:
            raise ValueError('complete resident audit files must be pinned')
    required=[root/n for n in ('launch.json','prepared.json','config.json','source-manifest.json','run/terminal.json','run/result.json')]
    if any(str(p) not in audit['files'] for p in required):raise ValueError('resident audit incomplete')
    if pinned(root/'run/terminal.json')['exit_code']!=0:raise ValueError('resident suite not complete')
    entry=[p for p in audit['policies'] if p['key']==binding['key']]
    if (len(entry)!=1 or entry[0].get('passed') is not True or entry[0]['requests']!=178
            or entry[0]['policy_id']!=policy_id or entry[0]['gate_root']!=config['acceptance_root']):
        raise ValueError('selected resident policy differs')
    gate=Path(config['acceptance_root'])
    if gate!=root/'run'/binding['key']:raise ValueError('resident gate path differs')
    for name in ('config.json','run/READY.json','run/terminal.json','run/hf/result.json','run/hf/terminal.json','run/vllm/result.json','run/vllm/terminal.json'):
        if str(gate/name) not in audit['files']:raise ValueError('selected policy audit incomplete')
    cfg=pinned(gate/'config.json')
    if (cfg['model']!=config['model_path'] or cfg['runtime_profile']!=config['runtime_profile']
            or cfg['base_id']+':'+cfg['adapter_id']!=policy_id):
        raise ValueError('resident behavior runtime differs')
    return dict(schema='jev-resident-behavior-consumption-v1',audit_sha256=file_hash(binding['audit']),
        key=binding['key'],container_id=audit['container_id'],policy_id=policy_id,
        scope='Pinned host inference acceptance only; optimizer/state lineage requires separate admission.')
