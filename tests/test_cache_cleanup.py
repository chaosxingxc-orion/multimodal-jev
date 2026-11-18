import json
import os
from pathlib import Path

import pytest

from tools import cache_cleanup as cleanup


def setup(tmp_path):
    root = tmp_path / 'project'; root.mkdir()
    (root / 'example.py').write_text('pass\n')
    cache = root / '__pycache__'; cache.mkdir()
    bytecode = cache / 'example.cpython-311.pyc'; bytecode.write_bytes(b'cached')
    tests = root / '.pytest_cache'; tests.mkdir()
    (tests / 'nodeids').write_text('[]')
    (root / 'checkpoint.pt').write_bytes(b'keep')
    config = dict(schema='local-cache-cleanup-v1', roots=[str(root)], protected_paths=[],
                  pin_manifests=[], min_age_seconds=0, max_files=100, max_bytes=10000,
                  lock_path=str(tmp_path / 'locks/cleanup.lock'), output_directory=str(tmp_path / 'output'))
    return config, root, bytecode


def test_only_regenerable_files_selected(tmp_path):
    config, root, bytecode = setup(tmp_path)
    (root / '__pycache__/orphan.cpython-311.pyc').write_bytes(b'no source')
    assert {Path(r['path']).name for r in cleanup.plan(config)['files']} == {bytecode.name, 'nodeids'}
    config['min_age_seconds'] = 3600
    assert cleanup.plan(config)['files'] == []


def test_pinned_sources_and_protected_caches_preserved(tmp_path):
    config, root, _ = setup(tmp_path)
    pins = tmp_path / 'pins.json'
    pins.write_text(json.dumps({'pins': {str(root / 'example.py'): cleanup.sha(root / 'example.py')}}))
    config['pin_manifests'] = [str(pins)]
    config['protected_paths'] = [str(root / '.pytest_cache')]
    assert cleanup.plan(config)['files'] == []


def test_symlinks_hardlinks_and_excluded_trees_preserved(tmp_path):
    config, root, bytecode = setup(tmp_path)
    os.link(bytecode, root / 'linked')
    (root / '.pytest_cache/link').symlink_to(root / 'example.py')
    hidden = root / 'datasets/.pytest_cache'; hidden.mkdir(parents=True)
    (hidden / 'cache').write_text('keep')
    assert [Path(r['path']).name for r in cleanup.plan(config)['files']] == ['nodeids']


def test_budget_exceeded_before_deletion(tmp_path):
    config, root, bytecode = setup(tmp_path)
    config['max_files'] = 1
    with pytest.raises(ValueError, match='budget'):
        cleanup.plan(config)
    assert bytecode.exists()


def test_replaced_file_rejected(tmp_path):
    config, root, bytecode = setup(tmp_path)
    row = next(r for r in cleanup.plan(config)['files'] if r['path'] == str(bytecode))
    bytecode.write_bytes(b'changed')
    with pytest.raises(ValueError, match='identity changed'):
        cleanup.delete_one(row, [])
    assert bytecode.exists()


def test_parent_symlink_swap_rejected(tmp_path):
    config, root, bytecode = setup(tmp_path)
    row = next(r for r in cleanup.plan(config)['files'] if r['path'] == str(bytecode))
    cache = bytecode.parent
    moved = root / 'moved'; cache.rename(moved); cache.symlink_to(moved)
    with pytest.raises(ValueError, match='path/protection'):
        cleanup.delete_one(row, [])
    assert (moved / bytecode.name).exists()


def test_apply_journals_and_idempotence(tmp_path, monkeypatch):
    config, root, bytecode = setup(tmp_path)
    template = tmp_path / 'config.json'; template.write_text(json.dumps(config))
    local = tmp_path / 'local.json'; local.write_text(json.dumps({'schema': 'jev-local-macros-v1', 'macros': {}}))
    monkeypatch.setattr(cleanup, 'check_paths', lambda *a: [])
    result = cleanup.run(template, local, apply=True)
    assert result['deleted_files'] == 2
    assert (root / 'example.py').exists() and (root / 'checkpoint.pt').exists()
    assert len(list((tmp_path / 'output/events').glob('*-deleted.json'))) == 2
    assert cleanup.run(template, local, apply=True)['status'] == 'already_complete'


def test_failed_cleanup_not_retried(tmp_path, monkeypatch):
    config, root, bytecode = setup(tmp_path)
    template = tmp_path / 'config.json'; template.write_text(json.dumps(config))
    local = tmp_path / 'local.json'; local.write_text(json.dumps({'schema': 'jev-local-macros-v1', 'macros': {}}))
    monkeypatch.setattr(cleanup, 'check_paths', lambda *a: [])
    def fail(*a): raise ValueError('changed file')
    monkeypatch.setattr(cleanup, 'delete_one', fail)
    with pytest.raises(ValueError, match='changed file'): cleanup.run(template, local, True)
    assert not (tmp_path / 'output/COMPLETE.json').exists()
    assert cleanup.read(tmp_path / 'output/status.json')['phase'] == 'failed_preserve_evidence'
    with pytest.raises(ValueError, match='Incomplete cleanup'): cleanup.run(template, local, True)
    assert bytecode.exists()


def test_plan_only_never_deletes(tmp_path, monkeypatch):
    config, root, bytecode = setup(tmp_path)
    template = tmp_path / 'config.json'; template.write_text(json.dumps(config))
    local = tmp_path / 'local.json'; local.write_text(json.dumps({'schema': 'jev-local-macros-v1', 'macros': {}}))
    monkeypatch.setattr(cleanup, 'check_paths', lambda *a: [])
    assert cleanup.run(template, local)['deleted_files'] == 0
    assert bytecode.exists()
    with pytest.raises(ValueError, match='Existing run differs'): cleanup.run(template, local, True)
