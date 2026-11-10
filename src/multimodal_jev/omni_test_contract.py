"""Full official FSC TEST admission after completed, frozen TRAIN selection."""
from collections import Counter
import json
from pathlib import Path

from .baseline import file_hash
from .omni_selection_contract import actor_config

RELEASE = 'e046b9e58357e26343b628adce1bcbe178472fa2de0290b6480d489c52fc527c'
REQUEST_SHA = '796ee08f908eaecd7b5e92686171a85e1b0f9c36ad35941858860492715f04fd'
TARGET_SHA = '66125831109ffe3471042ff43596b7e42fd5410874d74c47e489be25e0a3e875'
GROUPS = 3793


def test_actor_config(cfg):
    result = actor_config(cfg)
    for name in ('test_protocol', 'test_protocol_sha256'):
        result.pop(name, None)
    return result


def select_test(cfg, rows):
    if (cfg['split'] != 'test' or cfg.get('evaluation_phase') != 'test'
            or cfg['audio_condition'] != 'actual' or cfg['parameters_frozen'] is not True
            or cfg.get('source_groups') is not None or cfg.get('adapter') is not None):
        raise ValueError('full frozen actual TEST only')
    path = Path(cfg['test_protocol'])
    if file_hash(path) != cfg['test_protocol_sha256']:
        raise ValueError('TEST protocol changed')
    protocol = json.loads(path.read_text())
    if (protocol['schema'] != 'jev-omni-test-protocol-v1'
            or protocol['actor_config'] != test_actor_config(cfg)
            or protocol['test_feedback_used'] is not False):
        raise ValueError('registered TEST actor changed')
    for name in ('selection_result', 'selection_protocol'):
        if file_hash(protocol[name]) != protocol[name+'_sha256']:
            raise ValueError('frozen selection evidence changed')
    selected = json.loads(Path(protocol['selection_result']).read_text())
    prior = json.loads(Path(protocol['selection_protocol']).read_text())
    winner = selected['selected_id']
    expected = {'parent': prior['prompts']['parent']}
    if winner != 'parent':
        expected[winner] = prior['prompts'][winner]
    if (selected['schema'] != 'jev-omni-prompt-selection-v1'
            or selected['protocol_sha256'] != protocol['selection_protocol_sha256']
            or selected['test_feedback_used'] is not False
            or selected['weight_updates'] != 0
            or selected['selected_instruction_sha256'] != expected[winner]
            or protocol['prompts'] != expected
            or file_hash(cfg['instruction']) not in expected.values()):
        raise ValueError('only parent and final selected prompt permitted')
    if file_hash(cfg['asset_audit']) != protocol['asset_audit_sha256']:
        raise ValueError('model asset audit changed')
    requests = Path(cfg['requests'])
    manifest = json.loads((requests.parent/'MANIFEST.json').read_text())
    pins = manifest['splits']['test']
    if (requests.name != 'test.requests.jsonl' or manifest['release_id'] != RELEASE
            or pins['requests_sha256'] != REQUEST_SHA or file_hash(requests) != REQUEST_SHA
            or pins['targets_sha256'] != TARGET_SHA or pins['source_groups'] != GROUPS
            or pins['requests'] != GROUPS*6 or protocol['requests_sha256'] != REQUEST_SHA
            or protocol['targets_sha256'] != TARGET_SHA):
        raise ValueError('pinned complete official TEST release required')
    counts = Counter(row['source_group'] for row in rows)
    if (len(rows) != GROUPS*6 or len({r['id'] for r in rows}) != GROUPS*6
            or len(counts) != GROUPS or set(counts.values()) != {6}
            or any(r['split'] != 'test' for r in rows)):
        raise ValueError('all3793 TEST recordings and six views required')
    return rows
