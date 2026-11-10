"""Pinned native SLURP full TEST, retaining the pre-existing composite prompt."""
import json
from pathlib import Path

from .baseline import file_hash

RELEASE='240070cb69d71f5df53874ff9e728611dd1bd56a564f7f9e7718092125612add'
REQUEST_SHA='69f871d49407a6dbd10cb5cb24ad712497880796c92d41a83424ac878ff78fe4'
TARGET_SHA='a88ddd796d3e4943e7964c5926b6333f12bc3f0bb238621a3bfa061e031d119a'


def select_native_test(cfg,rows):
    if (cfg['split']!='test' or cfg['task_format']!='slurp-native'
            or cfg['evaluation_phase']!='slurp-native-test' or cfg['audio_condition']!='actual'
            or cfg['parameters_frozen'] is not True or cfg['max_new_tokens']!=256
            or cfg.get('source_groups') is not None):
        raise ValueError('complete frozen native SLURP TEST required')
    reference=Path(cfg['population_reference'])
    if file_hash(reference)!=cfg['population_reference_sha256']:
        raise ValueError('native baseline reference changed')
    prior=json.loads(reference.read_text())
    for key in ('audio_model','taxonomy_sha256','max_new_tokens','max_context_tokens',
                'seed','dtype','sampling','route','task_format'):
        if cfg[key]!=prior[key]:raise ValueError('frozen native baseline changed: '+key)
    if file_hash(cfg['instruction'])!=file_hash(prior['instruction']):
        raise ValueError('original native prompt required')
    requests=Path(cfg['requests']);manifest=json.loads((requests.parent/'MANIFEST.json').read_text())
    pins=manifest['splits']['test']
    if (requests.name!='test.requests.jsonl' or manifest['release_id']!=RELEASE
            or file_hash(requests)!=REQUEST_SHA or pins['requests_sha256']!=REQUEST_SHA
            or pins['targets_sha256']!=TARGET_SHA or pins['recordings']!=13078
            or file_hash(cfg['taxonomy'])!=cfg['taxonomy_sha256']
            or cfg['taxonomy_sha256']!=manifest['identity']['taxonomy_sha256']):
        raise ValueError('pinned full native TEST release required')
    if (len(rows)!=13078 or len({r['id'] for r in rows})!=13078
            or len({r['source_group'] for r in rows})!=2974
            or any(r['split']!='test' for r in rows)):
        raise ValueError('all13078 TEST recordings required')
    return rows
