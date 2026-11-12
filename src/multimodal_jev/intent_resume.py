"""Authenticate fixed-topology intent RL continuation before loading shards.

The dispatch controller must independently observe predecessor containers exit
zero. This contract authenticates their pinned accepted artifacts and schedule.
"""
import json
from pathlib import Path
from .baseline import file_hash


def validate_lineage(previous,current):
    for key in ('model_path','pilot','manifest_sha256','arm','global_requests','seed',
                'lr','lora_rank','lora_alpha','lora_execution_dtype','runtime_profile'):
        if key not in previous or previous[key]!=current.get(key):
            raise ValueError('intent continuation configuration changed: '+key)
    if (type(previous.get('step'))!=int or type(current.get('step'))!=int
            or previous['step']<0 or current['step']!=previous['step']+1):
        raise ValueError('intent continuation must advance exactly one global update')
    if (Path(current['initial_adapter'])!=Path(current['previous_root'])/'run/adapter'
            or Path(current['acceptance_root'])!=Path(current['previous_suite'])):
        raise ValueError('continuation requires accepted predecessor adapter and policy')


def authenticate(config):
    root=Path(config['previous_root']);suite=Path(config['previous_suite'])
    def pinned(path):
        if config['pins'].get(str(path))!=file_hash(path):
            raise ValueError('continuation evidence not pinned: '+str(path))
        return json.loads(path.read_text())
    previous=pinned(root/'config.json');validate_lineage(previous,config)
    accepted=pinned(suite/'ACCEPTED.json');verified=pinned(suite/'verified.json')
    if pinned(suite/'controller-terminal.json')['exit_code']!=0:
        raise ValueError('predecessor acceptance failed')
    gate=pinned(suite/'config.json')
    if (gate['training_root']!=str(root) or gate['adapter']!=config['initial_adapter']
            or gate['model']!=config['model_path'] or gate['runtime_profile']!=config['runtime_profile']):
        raise ValueError('predecessor suite lineage differs')
    if (accepted['passed'] is not True or verified['passed'] is not True
            or accepted['verified_sha256']!=file_hash(suite/'verified.json')
            or verified['root']!=str(root) or verified['suite']!=str(suite)
            or verified['updates']!=config['step'] or verified['arm']!=config['arm']
            or verified['full_gate_requests']!=178 or not verified['exact_export_reload']
            or accepted['adapter_id']!=verified['adapter_id'] or gate['adapter_id']!=verified['adapter_id']):
        raise ValueError('accepted predecessor update differs')
    for path,digest in verified['files'].items():
        if config['pins'].get(path)!=digest or file_hash(path)!=digest:
            raise ValueError('verified predecessor state changed or not pinned')
    for rank in range(4):
        path=root/'run'/f'rank-{rank}'/'optimizer-local.pt'
        if str(path) not in verified['files']:raise ValueError('all predecessor optimizer shards required')
    return dict(previous_root=str(root),previous_suite=str(suite),previous_updates=config['step'],
                adapter_id=verified['adapter_id'],verified_sha256=file_hash(suite/'verified.json'))
