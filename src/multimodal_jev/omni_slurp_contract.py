"""Reuse the original complete native SLURP TRAIN engineering population."""
import json
from pathlib import Path

from .baseline import file_hash


def select_native_probe(cfg, rows):
    if (cfg['split'] != 'train' or cfg['task_format'] != 'slurp-native'
            or cfg['audio_condition'] != 'actual' or cfg['parameters_frozen'] is not True
            or cfg['max_new_tokens'] != 256):
        raise ValueError('native TRAIN frozen actual probe with 256 output tokens required')
    prior = Path(cfg['population_reference'])
    if file_hash(prior) != cfg['population_reference_sha256']:
        raise ValueError('registered native population reference changed')
    reference = json.loads(prior.read_text())
    for key in ('requests', 'taxonomy', 'taxonomy_sha256', 'source_groups', 'split', 'task_format'):
        if cfg[key] != reference[key]:
            raise ValueError('original native population changed: '+key)
    if (len(cfg['source_groups']) != 16 or len(set(cfg['source_groups'])) != 16
            or file_hash(cfg['instruction']) != file_hash(reference['instruction'])):
        raise ValueError('original16 groups and native instruction required')
    release = Path(cfg['requests']).parent
    manifest = json.loads((release/'MANIFEST.json').read_text())
    if (file_hash(cfg['requests']) != manifest['splits']['train']['requests_sha256']
            or file_hash(cfg['taxonomy']) != manifest['identity']['taxonomy_sha256']
            or cfg['taxonomy_sha256'] != manifest['identity']['taxonomy_sha256']):
        raise ValueError('native release changed')
    groups = set(cfg['source_groups'])
    selected = [r for r in rows if r['source_group'] in groups]
    if (len(selected) != 60 or len({r['id'] for r in selected}) != 60
            or {r['source_group'] for r in selected} != groups
            or any(r['split'] != 'train' for r in selected)):
        raise ValueError('complete original60 TRAIN recordings required')
    return selected


def validate_runtime(cfg):
    """The engine below implements these exact parameters; reject misleading config."""
    if (cfg['dtype'] != 'bfloat16' or cfg['seed'] != 17
            or cfg['max_context_tokens'] != 4096
            or cfg['sampling'] != {'temperature':0., 'n':1}
            or cfg['max_new_tokens'] not in (128,256)):
        raise ValueError('registered BF16 greedy seed17 context4096 runtime required')
    if cfg['max_new_tokens'] == 256 and cfg.get('task_format') != 'slurp-native':
        raise ValueError('256 output tokens reserved for native SLURP probe')
