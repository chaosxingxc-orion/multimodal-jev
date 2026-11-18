import hashlib
import pytest

import multimodal_jev.local_audio_mirror as module
import multimodal_jev.local_checkpoint as checkpoint


def fixture(tmp_path, monkeypatch):
    # Storage admission has separate mount-namespace tests; these test byte identity.
    monkeypatch.setattr(module, 'check_paths', lambda *a, **k: None)
    monkeypatch.setattr(checkpoint, 'check_paths', lambda *a, **k: None)
    source = tmp_path / 'source.wav'
    source.write_bytes(b'admitted audio')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return module.LocalAudioMirror(tmp_path / 'mirror'), source, digest


def test_replica_survives_source_removal_and_rejects_corruption(tmp_path, monkeypatch):
    mirror, source, digest = fixture(tmp_path, monkeypatch)
    local = mirror.publish(source, digest)
    source.unlink()
    assert mirror.publish(source, digest) == local
    assert mirror.resolve(digest).read_bytes() == b'admitted audio'
    local.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='changed'):
        mirror.resolve(digest)
    with pytest.raises(ValueError, match='changed'):
        mirror.publish(source, digest)


def test_wrong_source_never_published(tmp_path, monkeypatch):
    mirror, source, digest = fixture(tmp_path, monkeypatch)
    source.write_bytes(b'changed source')
    with pytest.raises(ValueError, match='source audio differs'):
        mirror.publish(source, digest)
    assert not (mirror.root / (digest + '.audio')).exists()
    assert sorted(p.name for p in mirror.root.iterdir()) == ['.publish.lock']


def test_missing_symlink_and_unsafe_identity_rejected(tmp_path, monkeypatch):
    mirror, source, digest = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='missing'):
        mirror.resolve(digest)
    (mirror.root / (digest + '.audio')).symlink_to(source)
    with pytest.raises(ValueError, match='changed'):
        mirror.publish(source, digest)
    with pytest.raises(ValueError, match='SHA256'):
        mirror.resolve('../source.wav')


def test_read_only_mirror_never_publishes_and_rejects_shared_mount(tmp_path,monkeypatch):
    mirror,source,digest=fixture(tmp_path,monkeypatch)
    local=mirror.publish(source,digest)
    monkeypatch.setattr(module,'mount_for',lambda *a:dict(filesystem='ext4'))
    reader=module.LocalAudioMirror(mirror.root,read_only=True)
    assert reader.resolve(digest)==local
    with pytest.raises(ValueError,match='read-only'):
        reader.publish(source,digest)
    monkeypatch.setattr(module,'mount_for',lambda *a:dict(filesystem='ceph'))
    with pytest.raises(ValueError,match='existing local'):
        module.LocalAudioMirror(mirror.root,read_only=True)
