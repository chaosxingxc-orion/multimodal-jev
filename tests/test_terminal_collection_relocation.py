import fcntl
import json

import pytest

from tools.reports import relocate_terminal_collection as relocation


def fixture(tmp_path, monkeypatch):
    source = tmp_path/'original'/'collected'
    source.mkdir(parents=True)
    (source/'nested').mkdir()
    (source/'nested'/'checkpoint.bin').write_bytes(b'unchanged checkpoint')
    archive = source.with_suffix('.tar')
    archive.write_bytes(b'archive bytes')
    files = relocation.inventory(source)
    transport = dict(status='transport_complete_independent_audit_required', files=files,
        container_id='terminal', state=dict(Status='exited', ExitCode=0, OOMKilled=False),
        archive_sha256=relocation.sha(archive))
    proof = dict(passed=True, files=files, container_id='terminal', state=transport['state'])
    config = dict(schema='terminal-collection-relocation-v1', collection=str(source),
        owner_lock=str(tmp_path/'owner.lock'), destination=str(tmp_path/'persistent'/'run'),
        evidence=str(tmp_path/'evidence'))
    (tmp_path/'owner.lock').touch()
    for name, data in [('transport', transport), ('proof', proof)]:
        path = tmp_path/(name+'.json')
        path.write_text(json.dumps(data))
        config[name] = str(path)
        config[name+'_sha256'] = relocation.sha(path)
    monkeypatch.setattr(relocation, 'check_paths', lambda _: [dict(free_bytes=10**12, free_fraction=.9)])
    monkeypatch.setattr(relocation, 'mount_for', lambda *args: dict(filesystem='ext4', options='rw'))
    return source, archive, config


def test_complete_retains_bytes_and_old_paths(tmp_path, monkeypatch):
    source, archive, config = fixture(tmp_path, monkeypatch)
    before = relocation.inventory(source)
    relocation.relocate(config)
    assert source.is_symlink() and archive.is_symlink()
    assert relocation.inventory(source) == before
    assert archive.read_bytes() == b'archive bytes'
    assert not source.with_name('collected.relocation-original').exists()
    assert (tmp_path/'evidence'/'COMPLETE.json').exists()
    with pytest.raises(ValueError):
        relocation.relocate(config)


@pytest.mark.parametrize('fault', ['changed_source', 'extra_file', 'symlink', 'changed_archive', 'identity', 'live_lock', 'no_capacity'])
def test_refusal_keeps_original(tmp_path, monkeypatch, fault):
    source, archive, config = fixture(tmp_path, monkeypatch)
    owner = None
    if fault == 'changed_source': (source/'nested/checkpoint.bin').write_bytes(b'changed')
    if fault == 'extra_file': (source/'unbound').touch()
    if fault == 'symlink': (source/'link').symlink_to(archive)
    if fault == 'changed_archive': archive.write_bytes(b'changed')
    if fault == 'identity': config['proof_sha256'] = '0'*64
    if fault == 'no_capacity':
        monkeypatch.setattr(relocation, 'check_paths', lambda _: [dict(free_bytes=20*1024**3, free_fraction=.1)])
    if fault == 'live_lock':
        owner = open(config['owner_lock'])
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises((ValueError, BlockingIOError)):
            relocation.relocate(config)
    finally:
        if owner: owner.close()
    assert source.is_dir() and not source.is_symlink()
    assert archive.is_file() and not archive.is_symlink()
    assert not (tmp_path/'evidence'/'COMPLETE.json').exists()


def test_failed_copy_preserves_original(tmp_path, monkeypatch):
    source, archive, config = fixture(tmp_path, monkeypatch)
    def fail(*args, **kwargs): raise OSError('disk write failed')
    monkeypatch.setattr(relocation, 'copy_durable', fail)
    with pytest.raises((OSError, relocation.shutil.Error)):
        relocation.relocate(config)
    assert not source.is_symlink() and archive.read_bytes() == b'archive bytes'
    assert (source/'nested/checkpoint.bin').read_bytes() == b'unchanged checkpoint'
