"""Build additive shared asset views and fetch pinned, checksum-verified files.

Existing files are never overwritten. Credentials are read by requests from the
process environment and are never included in receipts or exception messages.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import ssl
import threading
import time

import requests

# Use the OS trust store, including the environment's installed proxy CA.
os.environ.setdefault("REQUESTS_CA_BUNDLE", ssl.get_default_verify_paths().cafile or requests.certs.where())
_local = threading.local()


def http_get(url, **kwargs):
    """Reuse TLS connections within each worker; never persist headers/tokens."""
    if not hasattr(_local, 'session'):
        _local.session = requests.Session()
    return _local.session.get(url, **kwargs)


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def digest(path, algorithm, size):
    h = hashlib.new('sha1' if algorithm == 'git-sha1' else algorithm)
    if algorithm == 'git-sha1':
        h.update(f'blob {size}\0'.encode())
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def hardlink_view(source, target, journal):
    counts = {'linked': 0, 'same_inode': 0, 'conflicts': 0, 'errors': 0, 'bytes_linked': 0}
    source, target = Path(source), Path(target)
    def error(exc):
        counts['errors'] += 1
        journal.write(json.dumps({'error': type(exc).__name__, 'path': exc.filename}) + '\n')
    for current, dirs, files in os.walk(source, followlinks=False, onerror=error):
        dest = target / Path(current).relative_to(source)
        try:
            if dest.is_symlink():
                raise ValueError('destination directory is a symlink')
            dest.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as exc:
            error(exc) if isinstance(exc, OSError) else journal.write(str(exc) + '\n')
            dirs[:] = []
            continue
        for name in list(dirs):
            if (Path(current) / name).is_symlink():
                files.append(name)
                dirs.remove(name)
        for name in files:
            src, dst = Path(current) / name, dest / name
            try:
                st = src.lstat()
                if dst.exists() or dst.is_symlink():
                    ds = dst.lstat()
                    if (st.st_dev, st.st_ino) == (ds.st_dev, ds.st_ino):
                        counts['same_inode'] += 1
                    elif src.is_symlink() and dst.is_symlink() and src.resolve() == dst.resolve():
                        counts['same_inode'] += 1
                    else:
                        counts['conflicts'] += 1
                        journal.write(json.dumps({'conflict': str(dst), 'source': str(src)}) + '\n')
                    continue
                if stat.S_ISREG(st.st_mode):
                    os.link(src, dst)
                    counts['bytes_linked'] += st.st_size
                elif stat.S_ISLNK(st.st_mode):
                    # Absolute referent preserves the meaning of a relative source link.
                    dst.symlink_to(src.resolve(strict=False))
                else:
                    continue
                counts['linked'] += 1
            except OSError as exc:
                error(exc)
    return counts


def organize(root, state):
    root, state = Path(root), Path(state)
    result = []
    with (state / 'organize-events.jsonl').open('a') as journal:
        for category in ['models', 'datasets']:
            source = root / 'omni' / category
            for child in sorted(source.iterdir()):
                if not child.is_dir():
                    continue
                target = root / category / child.name
                counts = hardlink_view(child, target, journal)
                result.append({'source': str(child), 'target': str(target), **counts})
                save(state / 'organize-progress.json', result)
                print(json.dumps(result[-1]), flush=True)
    save(state / 'organize-terminal.json', {'exit_code': 0 if not any(r['errors'] or r['conflicts'] for r in result) else 1, 'results': result})


def fetch_one(item, root, state):
    dest = root / item['path']
    if not dest.resolve().is_relative_to(root.resolve()):
        raise ValueError('path outside asset root')
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected_size, expected_hash = item['size'], item['hash']
    algorithm = item['algorithm']
    receipt_key = hashlib.sha256(item['path'].encode()).hexdigest()
    receipt_path = state / 'receipts' / (receipt_key + '.json')
    # A persisted receipt is reusable only for the same inode, size, mtime and identity.
    if dest.exists() and receipt_path.exists():
        rec = json.loads(receipt_path.read_text())
        st = dest.stat()
        # Small files may be rewritten within one filesystem timestamp tick.
        # Rehash them; cache large immutable artifacts only after a quiet window.
        if expected_size > 1024 * 1024 and time.time() - st.st_mtime > 2 and rec.get('hash') == expected_hash and rec.get('size') == st.st_size == expected_size and rec.get('mtime_ns') == st.st_mtime_ns and rec.get('ctime_ns') == st.st_ctime_ns and rec.get('inode') == st.st_ino:
            return {'path': item['path'], 'status': 'verified_cached', 'size': expected_size}
    if dest.exists():
        if dest.stat().st_size != expected_size or digest(dest, algorithm, expected_size) != expected_hash:
            return {'path': item['path'], 'status': 'conflict', 'size': expected_size}
        status = 'verified_existing'
    else:
        if shutil.disk_usage(root).free < 100 * 1024 ** 3 + expected_size:
            return {'path': item['path'], 'status': 'insufficient_space', 'size': expected_size}
        part = dest.with_name(dest.name + '.download-part')
        # Chunk files are private state; never mutate a linked existing artifact.
        chunk_size = 16 * 1024 * 1024
        use_ranges = expected_size >= 128 * 1024 * 1024
        if use_ranges:
            progress_path = state / 'ranges' / (receipt_key + '.json')
            completed = set()
            if progress_path.exists() and part.exists():
                info = json.loads(progress_path.read_text())
                if info.get('hash') == expected_hash and part.stat().st_size == expected_size:
                    completed = set(info['completed'])
            fd = os.open(part, os.O_RDWR | os.O_CREAT, 0o664)
            os.ftruncate(fd, expected_size)
            lock = threading.Lock()
            def chunk(index):
                start, end = index * chunk_size, min((index + 1) * chunk_size, expected_size) - 1
                for attempt in range(4):
                    try:
                        with http_get(item['url'], headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'}, stream=True, timeout=(20, 90)) as r:
                            if r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{expected_size}':
                                raise ValueError('range not honored')
                            pos = start
                            for data in r.iter_content(1024 * 1024):
                                if pos + len(data) > end + 1:
                                    raise ValueError('oversize range')
                                view = memoryview(data)
                                while view:
                                    written = os.pwrite(fd, view, pos)
                                    view = view[written:]
                                    pos += written
                            if pos != end + 1:
                                raise ValueError('short range')
                        os.fsync(fd)
                        with lock:
                            completed.add(index)
                            save(progress_path, {'hash': expected_hash, 'completed': sorted(completed)})
                        return
                    except Exception:
                        if attempt == 3:
                            raise RuntimeError('range_download_failed') from None
                        time.sleep(2 ** attempt)
            try:
                pending = [i for i in range((expected_size + chunk_size - 1) // chunk_size) if i not in completed]
                chunk_workers = 64 if expected_size > 20 * 1024 ** 3 else 8
                with concurrent.futures.ThreadPoolExecutor(max_workers=chunk_workers) as pool:
                    for _ in pool.map(chunk, pending):
                        pass
            finally:
                os.close(fd)
        else:
            for attempt in range(4):
                try:
                    with http_get(item['url'], stream=True, timeout=(20, 90)) as r:
                        r.raise_for_status()
                        size = 0
                        with part.open('wb') as f:
                            for data in r.iter_content(1024 * 1024):
                                size += len(data)
                                if size > expected_size:
                                    raise ValueError('oversize download')
                                f.write(data)
                            f.flush()
                            os.fsync(f.fileno())
                    if size != expected_size:
                        raise ValueError('short download')
                    break
                except Exception:
                    if attempt == 3:
                        raise RuntimeError('download_failed') from None
                    time.sleep(2 ** attempt)
        if digest(part, algorithm, expected_size) != expected_hash:
            return {'path': item['path'], 'status': 'checksum_failed', 'size': expected_size}
        # link() refuses to clobber a file created by another process mid-download.
        os.link(part, dest)
        part.unlink()
        status = 'downloaded_verified'
    st = dest.stat()
    save(receipt_path, {'path': item['path'], 'size': expected_size, 'hash': expected_hash, 'algorithm': algorithm, 'mtime_ns': st.st_mtime_ns, 'ctime_ns': st.st_ctime_ns, 'inode': st.st_ino, 'verified_at': time.time()})
    return {'path': item['path'], 'status': status, 'size': expected_size}


def download(manifest, root, state, workers):
    items = json.loads(Path(manifest).read_text())['files']
    if len({i['path'] for i in items}) != len(items):
        raise ValueError('duplicate target paths')
    root, state = Path(root), Path(state)
    totals = {'total_files': len(items), 'total_bytes': sum(i['size'] for i in items), 'done': 0, 'verified_bytes': 0, 'errors': 0, 'started': time.time()}
    running = set()
    stop = threading.Event()
    lock = threading.Lock()
    def heartbeat():
        while not stop.wait(20):
            with lock:
                save(state / 'download-progress.json', {**totals, 'active_files': sorted(running), 'updated': time.time()})
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    def task(item):
        with lock:
            running.add(item['path'])
        try:
            return fetch_one(item, root, state)
        except Exception as exc:
            return {'path': item['path'], 'status': 'failed', 'error': type(exc).__name__, 'size': item['size']}
        finally:
            with lock:
                running.discard(item['path'])
    try:
        with (state / 'download-events.jsonl').open('a') as log:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(task, item) for item in items]
                for future in concurrent.futures.as_completed(futures):
                    rec = future.result()
                    log.write(json.dumps(rec) + '\n')
                    log.flush()
                    with lock:
                        totals['done'] += 1
                        if rec['status'] in ['downloaded_verified', 'verified_existing', 'verified_cached']:
                            totals['verified_bytes'] += rec['size']
                        else:
                            totals['errors'] += 1
                    if totals['done'] % 50 == 0:
                        print(json.dumps(totals), flush=True)
    finally:
        stop.set()
        thread.join()
        save(state / 'download-progress.json', {**totals, 'active_files': sorted(running), 'updated': time.time()})
    save(state / 'download-terminal.json', {**totals, 'exit_code': 0 if totals['errors'] == 0 else 1, 'finished': time.time()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['organize', 'download'])
    parser.add_argument('--root', required=True)
    parser.add_argument('--state', required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    state = Path(args.state)
    state.mkdir(parents=True, exist_ok=True)
    with (state / (args.mode + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == 'organize':
            organize(args.root, state)
        else:
            download(args.manifest, args.root, state, args.workers)


if __name__ == '__main__':
    main()
