"""Bounded deterministic file verification; no model or network dependency."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path


def workers(value):
    if type(value) is not int or not 1 <= value <= 32:
        raise ValueError('worker count must be an integer in [1, 32]')
    return value


def inventory(root, *, io_workers=4):
    """Read every byte on every call; never trust timestamps as integrity proof."""
    workers(io_workers)
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('ordinary evidence directory required')
    entries = sorted(root.rglob('*'))
    if any(p.is_symlink() for p in entries):
        raise ValueError('evidence symlinks forbidden')
    files = [p for p in entries if p.is_file()]
    def digest(path):
        result = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                result.update(block)
        return str(path.relative_to(root)), result.hexdigest()
    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        result = dict(pool.map(digest, files))
    # Detect additions/removals during this pass. Inputs must remain frozen;
    # this is not a filesystem snapshot or a concurrent-writer security claim.
    final = sorted(root.rglob('*'))
    if entries != final or any(p.is_symlink() for p in final):
        raise ValueError('evidence inventory changed during verification')
    return result
