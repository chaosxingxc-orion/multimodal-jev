"""Configured local regenerable-cache cleanup, without model or remote calls.

Only source-backed __pycache__ bytecode and pytest cache files are eligible.
No checkpoints, recursive directory removal, model/data caches or auto retries.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import time

from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.committed_signal_report import read, sha, write


EXCLUDED = {'.git', '.local', '.venv', 'venv', 'models', 'datasets', 'checkpoints'}


def identity(st):
    return {key: getattr(st, 'st_' + key) for key in
            ('dev', 'ino', 'size', 'mtime_ns', 'ctime_ns', 'nlink')}


def protected(path, paths):
    return any(path == p or path.is_relative_to(p) for p in paths)


def policy(config):
    required = {'schema', 'roots', 'protected_paths', 'pin_manifests', 'min_age_seconds',
                'max_files', 'max_bytes', 'lock_path', 'output_directory'}
    if set(config) != required or config['schema'] != 'local-cache-cleanup-v1':
        raise ValueError('Unknown cleanup policy schema or fields')
    for key in ('min_age_seconds', 'max_files', 'max_bytes'):
        if type(config[key]) is not int or config[key] < (0 if key == 'min_age_seconds' else 1):
            raise ValueError('Invalid limit: ' + key)
    if not isinstance(config['roots'], list) or not config['roots']:
        raise ValueError('Explicit nonempty roots required')
    roots = []
    for value in config['roots']:
        path = Path(value).absolute()
        if path.resolve() != path or not path.is_dir() or len(path.parts) < 3:
            raise ValueError('Existing non-symlink narrow roots required')
        if any(part in EXCLUDED for part in path.parts):
            raise ValueError('Protected root category')
        roots.append(path)
    if len(set(roots)) != len(roots) or any(a != b and a.is_relative_to(b) for a in roots for b in roots):
        raise ValueError('Cleanup roots must be disjoint')
    protection = [Path(p).resolve() for p in config['protected_paths']]
    protection += [Path(config[k]).resolve() for k in ('output_directory', 'lock_path')]
    manifest_hashes = {}
    for name in config['pin_manifests']:
        path = Path(name).resolve()
        data = read(path)
        pins = data.get('pins', data)
        if not isinstance(pins, dict) or not pins or any(not isinstance(v, str) or len(v) != 64 for v in pins.values()):
            raise ValueError('Expected a nonempty path-to-SHA256 pin map')
        protection.extend(Path(p).resolve() for p in pins)
        protection.append(path)
        manifest_hashes[str(path)] = sha(path)
    return roots, protection, manifest_hashes


def candidate(path, root, protection, cutoff):
    if protected(path, protection) or path.resolve() != path:
        return None
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or st.st_mtime > cutoff:
        return None
    relative = path.relative_to(root)
    if any(part in EXCLUDED for part in relative.parts):
        return None
    source = None
    if path.parent.name == '__pycache__' and path.suffix == '.pyc':
        try:
            source = Path(importlib.util.source_from_cache(str(path)))
        except ValueError:
            return None
        if (source.resolve() != source or not source.is_file()
                or protected(source, protection)):
            return None
        reason = 'regenerable_bytecode_with_source'
    elif '.pytest_cache' in relative.parts[:-1]:
        reason = 'regenerable_pytest_cache'
    else:
        return None
    row = dict(path=str(path), root=str(root), relative=str(relative),
               reason=reason, identity=identity(st), sha256=sha(path))
    if source is not None:
        row['source'] = str(source)
        row['source_sha256'] = sha(source)
    return row


def plan(config, now=None):
    roots, protection, manifests = policy(config)
    now = time.time() if now is None else now
    rows = []
    total = 0
    def fail(exc):
        raise exc
    for root in roots:
        for current, dirs, names in os.walk(root, followlinks=False, onerror=fail):
            dirs[:] = sorted(d for d in dirs if d not in EXCLUDED
                             and not (Path(current) / d).is_symlink()
                             and not protected(Path(current) / d, protection))
            for name in sorted(names):
                row = candidate(Path(current) / name, root, protection, now - config['min_age_seconds'])
                if row:
                    rows.append(row)
                    total += row['identity']['size']
                    if len(rows) > config['max_files'] or total > config['max_bytes']:
                        raise ValueError('Cleanup budget exceeded; no deletion allowed')
    return dict(schema='local-cache-cleanup-plan-v1', created_unix=now, files=rows,
                logical_bytes=total, pin_manifests=manifests,
                scope='Regenerable local caches only; no research artifacts or checkpoint deletion')


def delete_one(row, protection):
    """Open parent chain without symlinks; recheck identity before unlinkat."""
    root, path = Path(row['root']), Path(row['path'])
    if (path != root / row['relative'] or path.resolve() != path
            or not path.is_relative_to(root) or protected(path, protection)):
        raise ValueError('Candidate path/protection changed')
    if 'source' in row and (sha(row['source']) != row['source_sha256']
                            or protected(Path(row['source']), protection)):
        raise ValueError('Regeneration source changed')
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open('/', flags)
    try:
        for part in path.parent.parts[1:]:
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        file_fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
        try:
            st = os.fstat(file_fd)
            if not stat.S_ISREG(st.st_mode) or identity(st) != row['identity']:
                raise ValueError('Candidate identity changed')
            digest = hashlib.sha256()
            while chunk := os.read(file_fd, 1024 * 1024):
                digest.update(chunk)
            if digest.hexdigest() != row['sha256'] or identity(os.fstat(file_fd)) != row['identity']:
                raise ValueError('Candidate content changed')
            if identity(os.stat(path.name, dir_fd=fd, follow_symlinks=False)) != row['identity']:
                raise ValueError('Candidate replaced')
            os.unlink(path.name, dir_fd=fd)
            os.fsync(fd)
        finally:
            os.close(file_fd)
    finally:
        os.close(fd)


def run(template, local, apply=False):
    config, provenance = load_config(template, local)
    roots, protection, _ = policy(config)
    output, lock = Path(config['output_directory']).resolve(), Path(config['lock_path']).resolve()
    check_paths({**{f'root{i}': p for i, p in enumerate(roots)}, 'output': output, 'lock': lock.parent})
    lock.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        complete = output / 'COMPLETE.json'
        if complete.exists():
            previous = read(complete)
            if previous['provenance'] != provenance or previous['apply'] != apply:
                raise ValueError('Existing run differs; use a new output version')
            if previous['plan_sha256'] != sha(output / 'plan.json'):
                raise ValueError('Published plan changed')
            return dict(status='already_complete', **previous['summary'])
        if (output / 'status.json').exists():
            raise ValueError('Incomplete cleanup preserved; automatic retry disabled')
        write(output / 'status.json', dict(phase='planning', apply=apply, provenance=provenance))
        deleted = 0
        try:
            proposed = plan(config)
            write(output / 'plan.json', proposed)
            if apply:
                write(output / 'status.json', dict(phase='deleting', apply=True, provenance=provenance))
                events = output / 'events'
                events.mkdir()
                for i, row in enumerate(proposed['files']):
                    current_config, current_provenance = load_config(template, local)
                    if current_config != config or current_provenance != provenance:
                        raise ValueError('Cleanup configuration changed')
                    _, current_protection, manifests = policy(config)
                    if manifests != proposed['pin_manifests']:
                        raise ValueError('Dependency manifests changed; preserve remaining files')
                    check_paths({'candidate_parent': Path(row['path']).parent})
                    write(events / f'{i:06d}-intent.json', row)
                    delete_one(row, current_protection)
                    deleted += 1
                    write(events / f'{i:06d}-deleted.json', dict(path=row['path'], status='deleted'))
            summary = dict(candidates=len(proposed['files']), deleted_files=deleted,
                           candidate_logical_bytes=proposed['logical_bytes'])
            write(complete, dict(provenance=provenance, apply=apply, summary=summary,
                                 plan_sha256=sha(output / 'plan.json')))
            write(output / 'status.json', dict(phase='complete', **summary))
            return dict(status='complete', **summary)
        except Exception as exc:
            write(output / 'status.json', dict(phase='failed_preserve_evidence', error=repr(exc),
                deleted_files=deleted, automatic_retry=False))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', default=os.environ.get('JEV_LOCAL_CONFIG', '.local/config/macros.json'))
    parser.add_argument('--apply', action='store_true', help='Delete validated cache candidates; default is plan only')
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.local, args.apply)))


if __name__ == '__main__':
    main()
