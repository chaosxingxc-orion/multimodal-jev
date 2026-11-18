import json
from pathlib import Path

import pytest

from multimodal_jev.local_config import load_config, resolve
from tools.verify_portable_training import verify
from tools.reports.committed_signal_report import sha


def test_shared_training_templates_have_no_machine_paths():
    for name in ('context-grpo-actor-v1.json', 'context-grpo-service-v1.json'):
        raw = (Path('configs/portable/training') / name).read_text()
        for prefix in ('/home/', '/mnt/', '/data/', '.local/', 'xingchao'):
            assert prefix not in raw
    actor = json.loads(Path('configs/portable/training/context-grpo-actor-v1.json').read_text())
    assert actor['training_dispatch_allowed'] == '${TRAINING_DISPATCH_ALLOWED}'
    assert (actor['global_batch'], actor['microbatch'], actor['algorithm']) == (128, 8, 'grpo_v1')
    sample = json.loads(Path('configs/local-macros.example.json').read_text())
    assert sample['macros']['TRAINING_DISPATCH_ALLOWED'] is False


def test_team_can_rebind_locations_without_changing_training_parameters():
    template = json.loads(Path('configs/portable/training/context-grpo-actor-v1.json').read_text())
    sample = json.loads(Path('configs/local-macros.example.json').read_text())['macros']
    left, _ = resolve(template, sample)
    sample.update(SHARED_ROOT='/partner/assets', TRAIN_ROOT='/partner/run')
    right, _ = resolve(template, sample)
    assert right['model'].startswith('/partner/assets/')
    assert right['schedule'] == '/partner/run/source/schedule'
    for name in ('global_batch', 'microbatch', 'algorithm', 'runtime_profile',
                 'binary_policy_arithmetic', 'behavior_bounds', 'stop_update'):
        assert left[name] == right[name]
    assert right['training_dispatch_allowed'] is False


def test_equivalence_rejects_source_and_algorithm_changes(tmp_path):
    templates = tmp_path / 'templates'; templates.mkdir()
    source = tmp_path / 'source'; source.mkdir()
    local = tmp_path / 'local.json'
    local.write_text(json.dumps({'schema': 'jev-local-macros-v1', 'macros': {'ROOT': '/run'}}))
    originals = {'config.json': {'training_dispatch_allowed': True, 'path': '/run', 'algorithm': 'grpo'},
                 'service-config.json': {'path': '/run'}}
    for name, value in originals.items():
        (source / name).write_text(json.dumps(value))
    for name, value in [('context-grpo-actor-v1.json', dict(training_dispatch_allowed=False, path='${ROOT}', algorithm='grpo')),
                        ('context-grpo-service-v1.json', dict(path='${ROOT}'))]:
        (templates / name).write_text(json.dumps(value))
    (templates / 'SOURCE.json').write_text(json.dumps(dict(schema='portable-training-template-source-v1',
        source_sha256={name: sha(source / name) for name in originals})))
    assert verify(templates, local, source)['passed']
    actor = templates / 'context-grpo-actor-v1.json'
    actor.write_text(actor.read_text().replace('grpo', 'rloo'))
    with pytest.raises(ValueError, match='differs'):
        verify(templates, local, source)
    (source / 'config.json').write_text('{}')
    with pytest.raises(ValueError, match='Frozen source changed'):
        verify(templates, local, source)
