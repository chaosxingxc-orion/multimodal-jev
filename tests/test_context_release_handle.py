import json
from pathlib import Path

import pytest

from multimodal_jev import context_release_handle as module
from multimodal_jev.context_release_handle import ReleaseHandle


def make_release(root, marker='one'):
    path = Path(root)/'release'
    path.mkdir()
    (path/'MANIFEST.json').write_text(json.dumps(dict(marker=marker)))
    return path


def fake_loader(calls):
    def load(root, manifest_sha256, role, purpose):
        calls.append((str(root), manifest_sha256, role, purpose))
        return ([dict(id='r1')], dict(schema=dict(candidates=[])),
                dict(schema='slurp-official-clean-intent-v2', role=role))
    return load


def test_second_load_reuses_authenticated_evidence(tmp_path, monkeypatch):
    release = make_release(tmp_path)
    digest = module.file_hash(release/'MANIFEST.json')
    calls = []
    monkeypatch.setattr('multimodal_jev.slurp_portable_release.load', fake_loader(calls))
    handle = ReleaseHandle()
    data, task, manifest, reused = handle.load(release, digest, 'learning', 'optimize')
    assert reused is False and data == [dict(id='r1')]
    assert manifest['role'] == 'learning'
    again, _, _, reused_again = handle.load(release, digest, 'learning', 'optimize')
    assert reused_again is True and again == data
    assert len(calls) == 1
    assert handle.stats()['executions'] == 1 and handle.stats()['hits'] == 1


def test_role_and_purpose_are_part_of_the_key(tmp_path, monkeypatch):
    release = make_release(tmp_path)
    digest = module.file_hash(release/'MANIFEST.json')
    calls = []
    monkeypatch.setattr('multimodal_jev.slurp_portable_release.load', fake_loader(calls))
    handle = ReleaseHandle()
    handle.load(release, digest, 'learning', 'optimize')
    _, _, _, reused = handle.load(release, digest, 'selection', 'evaluate')
    assert reused is False and len(calls) == 2


def test_changed_manifest_bytes_miss_the_key(tmp_path, monkeypatch):
    release = make_release(tmp_path)
    calls = []
    monkeypatch.setattr('multimodal_jev.slurp_portable_release.load', fake_loader(calls))
    handle = ReleaseHandle()
    first = module.file_hash(release/'MANIFEST.json')
    handle.load(release, first, 'learning', 'optimize')
    (release/'MANIFEST.json').write_text(json.dumps(dict(marker='two')))
    second = module.file_hash(release/'MANIFEST.json')
    assert second != first
    # A stale digest is refused outright rather than serving cached evidence.
    with pytest.raises(ValueError):
        handle.load(release, first, 'learning', 'optimize')
    _, _, _, reused = handle.load(release, second, 'learning', 'optimize')
    assert reused is False and len(calls) == 2


def test_reader_code_identity_is_part_of_the_key(tmp_path, monkeypatch):
    release = make_release(tmp_path)
    digest = module.file_hash(release/'MANIFEST.json')
    calls = []
    monkeypatch.setattr('multimodal_jev.slurp_portable_release.load', fake_loader(calls))
    first, second = ReleaseHandle(), ReleaseHandle()
    second.readers = dict(first.readers, **{'multimodal_jev/slurp_training.py': 'f'*64})
    first.load(release, digest, 'learning', 'optimize')
    _, _, _, reused = second.load(release, digest, 'learning', 'optimize')
    assert reused is False and len(calls) == 2


def test_returned_evidence_is_a_private_copy(tmp_path, monkeypatch):
    release = make_release(tmp_path)
    digest = module.file_hash(release/'MANIFEST.json')
    monkeypatch.setattr('multimodal_jev.slurp_portable_release.load', fake_loader([]))
    handle = ReleaseHandle()
    data, _, _, _ = handle.load(release, digest, 'learning', 'optimize')
    data[0]['id'] = 'tampered'
    assert handle.load(release, digest, 'learning', 'optimize')[0] == [dict(id='r1')]


def test_missing_manifest_rejected(tmp_path):
    with pytest.raises(ValueError):
        ReleaseHandle().load(tmp_path, 'a'*64, 'learning', 'optimize')


def test_reader_digests_cover_every_participating_module():
    digests = module.reader_digests()
    assert set(digests) == set(module.READERS)
    assert all(len(value) == 64 for value in digests.values())