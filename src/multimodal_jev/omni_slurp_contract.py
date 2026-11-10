"""Reuse the original complete native SLURP TRAIN engineering population."""
import json
from pathlib import Path

from .baseline import file_hash
from .slurp_jev_full import expand_bundle


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
    if (cfg['max_new_tokens'] == 256 and cfg.get('task_format') != 'slurp-native'
            and cfg.get('evaluation_phase') != 'slurp-jev-probe'):
        raise ValueError('256 output tokens reserved for registered SLURP probes')


def select_jev_probe(cfg, rows):
    if (cfg['split'] != 'train' or cfg['task_format'] != 'jev'
            or cfg['audio_condition'] != 'actual' or cfg['parameters_frozen'] is not True
            or cfg['evaluation_phase'] != 'slurp-jev-probe' or cfg['max_new_tokens'] != 256):
        raise ValueError('complete TRAIN SLURP-Jev frozen probe required')
    prior = Path(cfg['population_reference'])
    if file_hash(prior) != cfg['population_reference_sha256']:
        raise ValueError('SLURP-Jev population reference changed')
    reference = json.loads(prior.read_text())
    for key in ('requests','source_groups','task_format','split'):
        if cfg[key] != reference[key]:
            raise ValueError('original Jev probe changed: '+key)
    if (cfg['source_groups'] != ['slurp:12507']
            or file_hash(cfg['instruction']) != file_hash(reference['instruction'])):
        raise ValueError('original group and full multivalue instruction required')
    request_path = Path(cfg['requests'])
    materialized = json.loads((request_path.parent/'MANIFEST.json').read_text())
    release = Path(materialized['source_release'])
    manifest = json.loads((release/'MANIFEST.json').read_text())
    if (file_hash(request_path) != materialized['requests_sha256']
            or file_hash(release/'MANIFEST.json') != materialized['source_manifest_sha256']
            or file_hash(release/'task-bank.json') != manifest['identity']['task_bank_sha256']
            or file_hash(release/'train.bundles.jsonl') != manifest['splits']['train']['bundles_sha256']):
        raise ValueError('full Jev release or materialized requests changed')
    bank = json.loads((release/'task-bank.json').read_text())
    expected = []
    for line in (release/'train.bundles.jsonl').open():
        bundle = json.loads(line)
        if bundle['source_group'] in cfg['source_groups']:
            expected.extend(expand_bundle(bundle,bank))
    if (len(expected) != 336 or len({r['id'] for r in expected}) != 336 or rows != expected
            or any(r['split'] != 'train' for r in rows)):
        raise ValueError('all168 questions for both original recordings required')
    return rows
