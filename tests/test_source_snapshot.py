import os
from pathlib import Path
import pytest
from multimodal_jev.source_snapshot import freeze, verify


def sample(tmp_path):
    source = tmp_path / 'worktree'; source.mkdir()
    (source / 'module.py').write_text('original')
    binding = freeze(source, tmp_path / 'snapshot', patterns=['*.py'], io_workers=2)
    return source, Path(binding['root']), binding['manifest_sha256']


def test_worktree_changes_do_not_change_verified_snapshot(tmp_path):
    source, root, identity = sample(tmp_path)
    (source / 'module.py').write_text('new collaborator change')
    (source / 'other.py').write_text('new file')
    assert list(verify(root, manifest_sha256=identity, io_workers=2)) == ['module.py']
    assert (root / 'module.py').read_text() == 'original'


@pytest.mark.parametrize('change', ['same_stat', 'added', 'removed', 'symlink', 'manifest'])
def test_changed_snapshot_is_rejected(tmp_path, change):
    _, root, identity = sample(tmp_path)
    root.chmod(0o755)
    p = root / 'module.py'; p.chmod(0o644)
    if change == 'same_stat':
        st = p.stat(); p.write_text('mutation'); os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    elif change == 'added': (root / 'other.py').write_text('new')
    elif change == 'removed': p.unlink()
    elif change == 'symlink': p.unlink(); p.symlink_to(tmp_path / 'worktree/module.py')
    else:
        p = root / 'FILES.json'; p.chmod(0o644); p.write_text('{}')
    with pytest.raises(ValueError): verify(root, manifest_sha256=identity, io_workers=2)


def test_symlinked_parent_not_accepted(tmp_path):
    real = tmp_path / 'real'; real.mkdir(); (real / 'a.py').write_text('code')
    source = tmp_path / 'source'; source.mkdir(); (source / 'link').symlink_to(real)
    with pytest.raises(ValueError, match='symlink'):
        freeze(source, tmp_path / 'snapshot', patterns=['link/*.py'], io_workers=1)
