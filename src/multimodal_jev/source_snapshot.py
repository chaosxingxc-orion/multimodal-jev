"""Content-bound, isolated source snapshots for deterministic verification.

Outputs and temporary files must live outside the snapshot. This is source
identity, not dependency-environment, fixed-image or device admission.
"""
import hashlib
import json
from pathlib import Path

from .context_execution import map_io
from .local_checkpoint import commit_json
from .local_runtime_storage import check_paths


def regular(root, relative):
    part = Path(relative)
    if part.is_absolute() or '..' in part.parts or not part.parts:
        raise ValueError('relative snapshot file required')
    target = root / part
    if any(p.is_symlink() for p in (target, *target.parents)):
        raise ValueError('snapshot symlink forbidden: ' + str(target))
    if not target.is_file():
        raise ValueError('snapshot regular file required: ' + str(target))
    return target


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root, paths, *, io_workers):
    def one(name):
        return name, digest(regular(root, name))
    items, _ = map_io(one, sorted(paths), workers=io_workers)
    return dict(items)


def verify(root, *, manifest_sha256, io_workers):
    root = Path(root).absolute()
    manifest = regular(root, 'FILES.json')
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise ValueError('snapshot manifest changed')
    expected = json.loads(raw)
    actual_names = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('snapshot contains symlink')
        if path.is_file() and path != manifest:
            actual_names.add(str(path.relative_to(root)))
    if actual_names != set(expected):
        raise ValueError('snapshot membership changed')
    actual = inventory(root, expected, io_workers=io_workers)
    if actual != expected:
        raise ValueError('snapshot file content changed')
    return actual


def freeze(source, output, *, patterns, io_workers):
    source, output = Path(source).absolute(), Path(output).absolute()
    if (source.is_relative_to(output) or output.is_relative_to(source)
            or output.exists() or output.is_symlink()):
        raise ValueError('fresh snapshot disjoint from worktree required')
    if not patterns or any(Path(p).is_absolute() or '..' in Path(p).parts for p in patterns):
        raise ValueError('explicit relative source patterns required')
    def names():
        return sorted({str(p.relative_to(source)) for pattern in patterns
                       for p in source.glob(pattern) if p.is_file()})
    selected = names()
    if not selected or 'FILES.json' in selected:
        raise ValueError('nonempty source selection without reserved manifest required')
    before = inventory(source, selected, io_workers=io_workers)
    check_paths({'source_snapshot': output})
    output.mkdir(parents=True, exist_ok=False)
    for name in selected:
        raw = regular(source, name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != before[name]:
            raise ValueError('source changed while freezing: ' + name)
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    if names() != selected or inventory(source, selected, io_workers=io_workers) != before:
        raise ValueError('source changed during snapshot transaction')
    commit_json(output / 'FILES.json', before)
    identity = digest(output / 'FILES.json')
    verify(output, manifest_sha256=identity, io_workers=io_workers)
    for path in output.rglob('*'):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted(output.rglob('*'), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir():
            path.chmod(0o555)
    output.chmod(0o555)
    return dict(root=str(output), manifest_sha256=identity, files=len(before))
