"""Admit only fixed TRAIN learning populations and explicit waveform controls."""
import json
from pathlib import Path
from .baseline import file_hash


def select_learning(cfg, rows):
    if cfg['split'] != 'train' or cfg.get('audio_condition', 'actual') not in ('actual', 'zeros'):
        raise ValueError('TRAIN actual/zero diagnostics only')
    groups = cfg['source_groups']
    if len(groups) not in (1, 64) or len(set(groups)) != len(groups):
        raise ValueError('one or all64 fixed learning groups required')
    if len(groups) == 64:
        if file_hash(cfg['partition']) != cfg['partition_sha256']:
            raise ValueError('learning partition changed')
        partition = json.loads(Path(cfg['partition']).read_text())
        if groups != partition['learning'] or set(groups) & set(partition['selection']):
            raise ValueError('exact original learning groups required')
    selected = [r for r in rows if r['source_group'] in set(groups)]
    if (len(selected) != 6 * len(groups) or len({r['id'] for r in selected}) != len(selected)
            or {r['source_group'] for r in selected} != set(groups)
            or any(r['split'] != 'train' for r in selected)):
        raise ValueError('complete unique six-view learning population required')
    for group in groups:
        if sum(r['source_group'] == group for r in selected) != 6:
            raise ValueError('six views per group required')
    return selected
