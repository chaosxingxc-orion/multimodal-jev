"""Regression tests for shared asset integrity and non-clobber semantics."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('shared_assets', Path(__file__).parents[1] / 'tools/shared_assets.py')
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


def test_hardlink_view_preserves_conflicts(tmp_path):
    src, dst = tmp_path / 'source', tmp_path / 'target'
    src.mkdir(); dst.mkdir()
    (src / 'weights').write_bytes(b'original')
    (src / 'config').write_bytes(b'new')
    (dst / 'config').write_bytes(b'keep')
    log = io.StringIO()
    result = assets.hardlink_view(src, dst, log)
    assert result['linked'] == 1 and result['conflicts'] == 1
    assert (src / 'weights').stat().st_ino == (dst / 'weights').stat().st_ino
    assert (dst / 'config').read_bytes() == b'keep'


def test_download_rejects_escape_and_existing_mismatch(tmp_path):
    root = tmp_path / 'root'; root.mkdir()
    state = tmp_path / 'state'
    item = {'path': '../outside', 'size': 4, 'hash': hashlib.sha256(b'good').hexdigest(), 'algorithm': 'sha256'}
    with pytest.raises(ValueError, match='outside'):
        assets.fetch_one(item, root, state)
    item['path'] = 'model'
    (root / 'model').write_bytes(b'evil')
    assert assets.fetch_one(item, root, state)['status'] == 'conflict'
    assert (root / 'model').read_bytes() == b'evil'


def test_receipt_does_not_hide_modified_content(tmp_path):
    root = tmp_path / 'root'; root.mkdir()
    state = tmp_path / 'state'
    target = root / 'data'; target.write_bytes(b'good')
    item = {'path': 'data', 'size': 4, 'hash': hashlib.sha256(b'good').hexdigest(), 'algorithm': 'sha256'}
    assert assets.fetch_one(item, root, state)['status'] == 'verified_existing'
    target.write_bytes(b'evil')
    assert assets.fetch_one(item, root, state)['status'] == 'conflict'
