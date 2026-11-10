"""Fail closed on a full FSC-Jev TEST population and a frozen prompt decision."""
import json
from pathlib import Path

from .baseline import file_hash


def validate_population(cfg, rows):
    groups = set(cfg['source_groups'])
    if len(groups) != len(cfg['source_groups']) or not groups <= {r['source_group'] for r in rows}:
        raise ValueError('invalid source groups')
    selected = [r for r in rows if r['source_group'] in groups]
    if (not selected or len({r['id'] for r in selected}) != len(selected)
            or any(r['split'] != cfg['split'] for r in selected)):
        raise ValueError('unique single-split population required')
    if cfg['split'] == 'train':
        return selected
    if cfg['split'] != 'test' or cfg['audio_condition'] != 'actual':
        raise ValueError('only TRAIN probes or locked actual-audio TEST permitted')
    lock = json.loads(Path(cfg['heldout_lock']).read_text())
    if (lock['schema'] != 'jev-direct-heldout-lock-v1'
            or lock['test_feedback_used'] is not False
            or lock['source_selection_split'] != 'train'
            or cfg['purpose'] != 'direct-audio-locked-full-test'):
        raise ValueError('frozen TRAIN selection lock required')
    if (len(rows) != len(selected) or len(selected) != lock['requests']
            or len(groups) != lock['source_groups']
            or groups != {r['source_group'] for r in rows}
            or file_hash(cfg['requests']) != lock['requests_sha256']):
        raise ValueError('full locked TEST coverage required')
    if file_hash(cfg['instruction']) not in (lock['parent_prompt_sha256'], lock['selected_prompt_sha256']):
        raise ValueError('prompt outside frozen selection')
    ignored = {'split', 'requests', 'source_groups', 'instruction', 'purpose', 'wall_seconds', 'heldout_lock'}
    if {k: v for k, v in cfg.items() if k not in ignored} != lock['actor_config']:
        raise ValueError('locked actor configuration changed')
    return selected
