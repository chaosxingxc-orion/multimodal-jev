import json
import pytest

from multimodal_jev import local_checkpoint as checkpoint


@pytest.fixture
def local_disk(monkeypatch):
    # Test filesystem may be overlay; storage admission has separate mount tests.
    monkeypatch.setattr(checkpoint, 'check_paths', lambda paths: None)


def test_writer_failure_never_publishes(tmp_path, local_disk):
    dest = tmp_path / 'state.pt'
    def broken(stream):
        stream.write(b'partial')
        raise RuntimeError('interrupted serialization')
    with pytest.raises(RuntimeError):
        checkpoint.commit_file(dest, broken)
    assert list(tmp_path.iterdir()) == []


def test_file_sync_failure_never_publishes(tmp_path, local_disk, monkeypatch):
    def broken(fd):
        raise OSError('I/O error')
    monkeypatch.setattr(checkpoint.os, 'fsync', broken)
    with pytest.raises(OSError):
        checkpoint.commit_json(tmp_path / 'state.json', {'ready': True})
    assert list(tmp_path.iterdir()) == []


def test_sync_precedes_publication_and_directory_sync_follows(tmp_path, local_disk, monkeypatch):
    dest = tmp_path / 'state.json'
    visibility = []
    original = checkpoint.os.fsync
    def sync(fd):
        visibility.append(dest.exists())
        original(fd)
    monkeypatch.setattr(checkpoint.os, 'fsync', sync)
    checkpoint.commit_json(dest, {'update': 2})
    assert visibility == [False, True]
    assert json.loads(dest.read_text()) == {'update': 2}
    assert dest.stat().st_mode & 0o777 == 0o644
    with pytest.raises(FileExistsError):
        checkpoint.commit_json(dest, {'update': 3})


def test_reject_shared_before_writer(tmp_path, monkeypatch):
    def rejected(paths):
        raise ValueError('shared filesystem')
    monkeypatch.setattr(checkpoint, 'check_paths', rejected)
    with pytest.raises(ValueError, match='shared'):
        checkpoint.commit_json(tmp_path / 'state.json', {})
    assert list(tmp_path.iterdir()) == []


def test_adapter_seal_publishes_readable_unchanged_bytes(tmp_path, local_disk, monkeypatch):
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    names = ('adapter_model.safetensors', 'adapter_config.json')
    original = {}
    for name in names:
        path = adapter / name
        original[name] = name.encode()
        path.write_bytes(original[name])
        path.chmod(0o600)
    modes = []
    real_sync = checkpoint.os.fsync
    def sync(fd):
        modes.append(checkpoint.os.fstat(fd).st_mode & 0o777)
        real_sync(fd)
    monkeypatch.setattr(checkpoint.os, 'fsync', sync)
    checkpoint.seal_adapter(adapter)
    assert modes[:2] == [0o644, 0o644]
    assert all((adapter / name).read_bytes() == original[name] for name in names)
    assert all((adapter / name).stat().st_mode & 0o777 == 0o644 for name in names)
