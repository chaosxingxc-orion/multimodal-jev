import json
from pathlib import Path

import pytest

from tools import retain_experiment_artifacts as retention
from tools import run_operations as ops
from test_artifact_retention import fixture


TEMPLATE = 'configs/portable/operations/retention-pipeline.json'


def setup(tmp_path, monkeypatch):
    root, inventory, data, config = fixture(tmp_path)
    macros = dict(WORKSPACE=str(Path.cwd()), RETENTION_PIPELINE_OUTPUT=str(tmp_path/'workflow'))
    for key in ('root', 'inventory', 'inventory_sha256', 'pin_manifests',
                'protected_paths', 'output_directory', 'lock_path'):
        name = {'output_directory': 'OUTPUT', 'lock_path': 'LOCK'}.get(key, key.upper())
        macros['RETENTION_' + name] = config[key]
    local = tmp_path/'local.json'
    local.write_text(json.dumps(dict(schema='jev-local-macros-v1', macros=macros)))
    for module in (ops, retention):
        monkeypatch.setattr(module, 'check_paths', lambda *a: [])
    monkeypatch.setattr(retention, 'delete_one', lambda *a: pytest.fail('plan must never delete'))
    return root, local


def test_plan_and_reuse_preserve_candidates(tmp_path, monkeypatch):
    root, local = setup(tmp_path, monkeypatch)
    original = local.read_bytes()
    assert ops.run(TEMPLATE, local)['completed'] == ['retention']
    assert ops.run(TEMPLATE, local)['status'] == 'already_complete'
    assert local.read_bytes() == original
    manifest = ops.read(tmp_path/'workflow/COMPLETE.json')
    for name in ('checkpoint1.pt', 'analysis.json'):
        assert (root/name).exists()
        assert str(root/name) in manifest['sources']


@pytest.mark.parametrize('target', ['candidate', 'archive', 'plan'])
def test_reuse_rejects_drift(tmp_path, monkeypatch, target):
    root, local = setup(tmp_path, monkeypatch)
    ops.run(TEMPLATE, local)
    path = {'candidate': root/'analysis.json', 'archive': tmp_path/'archived-conclusion.md',
            'plan': tmp_path/'publication/plan.json'}[target]
    path.write_text('changed')
    with pytest.raises(ValueError, match='Source changed'):
        ops.run(TEMPLATE, local)


@pytest.mark.parametrize('fault', ['apply', 'deleted', 'outputs'])
def test_refuses_nonplan_publications(tmp_path, monkeypatch, fault):
    _, local = setup(tmp_path, monkeypatch)
    ops.run(TEMPLATE, local)
    complete = tmp_path/'publication/COMPLETE.json'
    data = ops.read(complete)
    if fault == 'apply': data['apply'] = True
    elif fault == 'deleted': data['deleted_paths'] = ['/unexpected']
    else: data['outputs'][str(tmp_path/'outside')] = '0' * 64
    complete.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='plan-only'):
        ops.verified_publication('retention_plan', complete)


def test_requires_v4_and_preserves_v3_collection():
    config = ops.read(TEMPLATE)
    config['schema'] = 'local-operations-v3'
    with pytest.raises(ValueError, match='v4'): ops.ordered_steps(config)
    config = ops.read('configs/portable/operations/collect-trajectory-pipeline.json')
    config['schema'] = 'local-operations-v4'
    assert len(ops.ordered_steps(config)) == 2
