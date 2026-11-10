"""Admit full held-out TRAIN selection only under a frozen prompt roster."""
import json
from pathlib import Path

from .baseline import file_hash


def actor_config(cfg):
    """Normalize mounted artifact paths while retaining their content identity."""
    result = dict(cfg)
    for name in ('instruction', 'partition', 'asset_audit', 'selection_protocol',
                 'selection_protocol_sha256'):
        result.pop(name, None)
    return result


def select_registered(cfg, rows):
    if (cfg['split'] != 'train' or cfg['audio_condition'] != 'actual'
            or cfg['parameters_frozen'] is not True
            or cfg.get('evaluation_phase') != 'selection'):
        raise ValueError('frozen actual TRAIN selection only')
    path = Path(cfg['selection_protocol'])
    if file_hash(path) != cfg['selection_protocol_sha256']:
        raise ValueError('selection protocol changed')
    protocol = json.loads(path.read_text())
    if (protocol['schema'] != 'jev-omni-selection-protocol-v1'
            or protocol['actor_config'] != actor_config(cfg)):
        raise ValueError('registered actor configuration changed')
    for path_key, hash_key in [('requests', 'requests_sha256'),
                               ('partition', 'partition_sha256'),
                               ('asset_audit', 'asset_audit_sha256')]:
        if file_hash(cfg[path_key]) != protocol[hash_key]:
            raise ValueError('registered input changed: ' + path_key)
    partition = json.loads(Path(cfg['partition']).read_text())
    learning, selection = partition['learning'], partition['selection']
    if (len(learning) != 64 or len(set(learning)) != 64
            or len(selection) != 256 or len(set(selection)) != 256
            or set(learning) & set(selection)
            or cfg['source_groups'] != selection
            or cfg['partition_sha256'] != protocol['partition_sha256']):
        raise ValueError('exact disjoint64/256 partition required')
    prompts = protocol['prompts']
    if (not 2 <= len(prompts) <= 3 or 'parent' not in prompts
            or not set(prompts) <= {'parent', 'p1', 'p2'}
            or len(set(prompts.values())) != len(prompts)
            or file_hash(cfg['instruction']) not in prompts.values()):
        raise ValueError('registered parent or reviewed candidate required')
    selected = [r for r in rows if r['source_group'] in set(selection)]
    if (len(selected) != 1536 or len({r['id'] for r in selected}) != 1536
            or any(r['split'] != 'train' for r in selected)
            or any(sum(r['source_group'] == g for r in selected) != 6 for g in selection)):
        raise ValueError('complete unique selection256 six-view population required')
    return selected
