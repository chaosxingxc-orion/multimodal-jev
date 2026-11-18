"""Relocate a closed, authenticated collection; retain original path aliases.

No remote calls, discovery, checkpoint selection, or deletion of unique data.
Interrupted migrations retain both copies and require explicit reconciliation.
"""
import argparse
import fcntl
import os
from pathlib import Path
import shutil

from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths, mount_for
from tools.jev.collect_context_rl_training import sha
from tools.reports.committed_signal_report import read, write


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def inventory(root):
    files = {}
    for p in root.rglob('*'):
        if p.is_symlink() or not (p.is_file() or p.is_dir()):
            raise ValueError('Only regular files and directories accepted')
        if p.is_file():
            files[str(p.relative_to(root))] = sha(p)
    return files


def copy_durable(source, target):
    source, target = Path(source), Path(target)
    with source.open('rb') as incoming, target.open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing, 8 * 1024**2)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    shutil.copystat(source, target)


def relocate(config):
    expected = {'schema', 'collection', 'transport', 'transport_sha256', 'proof',
                'proof_sha256', 'owner_lock', 'destination', 'evidence'}
    if set(config) != expected or config['schema'] != 'terminal-collection-relocation-v1':
        raise ValueError('Explicit terminal collection relocation required')
    source = Path(config['collection']).absolute()
    destination = Path(config['destination']).absolute()
    evidence = Path(config['evidence']).absolute()
    archive = source.with_suffix('.tar')
    backup = source.with_name(source.name + '.relocation-original')
    archive_backup = archive.with_name(archive.name + '.relocation-original')
    for p in (source, archive, destination, evidence):
        if p.resolve() != p:
            raise ValueError('Initial paths must not traverse symlinks')
    if any(p.exists() for p in (destination, evidence, backup, archive_backup)):
        raise ValueError('Existing relocation state; reconcile without retry')
    if (destination.is_relative_to(source) or source.is_relative_to(destination)
            or evidence.is_relative_to(source) or evidence.is_relative_to(destination)
            or source.is_relative_to(evidence) or destination.is_relative_to(evidence)):
        raise ValueError('Separate destination and evidence required')
    # Source is read-only until verified copies exist. Do not apply a free-space
    # admission threshold to removing redundant bytes from a full filesystem.
    mount = mount_for(source, Path('/proc/self/mountinfo').read_text())
    if mount['filesystem'] not in ('ext4', 'xfs', 'btrfs') or 'ro' in mount['options'].split(','):
        raise ValueError('Source must be writable local persistent storage')
    storage = check_paths({'destination': destination, 'evidence': evidence})
    with Path(config['owner_lock']).open('r') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ('transport', 'proof'):
            if sha(config[name]) != config[name + '_sha256']:
                raise ValueError('Identity changed: ' + name)
        transport, proof = read(config['transport']), read(config['proof'])
        state = transport['state']
        if (transport['status'] != 'transport_complete_independent_audit_required'
                or proof['passed'] is not True or proof['files'] != transport['files']
                or proof['container_id'] != transport['container_id']
                or proof['state'] != state or state['Status'] != 'exited'
                or state['ExitCode'] != 0 or state['OOMKilled']):
            raise ValueError('Audited successful terminal collection required')
        if inventory(source) != transport['files'] or sha(archive) != transport['archive_sha256']:
            raise ValueError('Source collection or archive changed')
        required = archive.stat().st_size + sum(p.stat().st_size for p in source.rglob('*') if p.is_file())
        free = storage[0]['free_bytes']
        total = free / storage[0]['free_fraction']
        if free - required < max(20 * 1024**3, total * .10):
            raise ValueError('Insufficient destination space after full copy')
        evidence.mkdir(parents=True)
        write(evidence/'intent.json', dict(config=config, bytes=required, storage=storage))
        destination.mkdir(parents=True)
        copied = destination/'collection'
        shutil.copytree(source, copied, copy_function=copy_durable)
        copied_archive = destination/'collection.tar'
        copy_durable(archive, copied_archive)
        if inventory(copied) != transport['files'] or sha(copied_archive) != transport['archive_sha256']:
            raise ValueError('Copied content mismatch; original retained')
        for p in sorted((p for p in copied.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            sync_dir(p)
        for p in (copied, destination, destination.parent):
            sync_dir(p)
        # Revalidate under the owner lock immediately before changing aliases.
        if inventory(source) != transport['files'] or sha(archive) != transport['archive_sha256']:
            raise ValueError('Source changed during copy; retain both copies')
        write(evidence/'verified.json', dict(files=transport['files'], archive_sha256=transport['archive_sha256']))
        source.rename(backup)
        source.symlink_to(copied, target_is_directory=True)
        archive.rename(archive_backup)
        archive.symlink_to(copied_archive)
        sync_dir(source.parent)
        write(evidence/'aliases.json', dict(collection=str(source.resolve()), archive=str(archive.resolve())))
        shutil.rmtree(backup)
        archive_backup.unlink()
        sync_dir(source.parent)
        write(evidence/'COMPLETE.json', dict(status='relocated_verified_original_paths_preserved',
              files=len(transport['files']), bytes=required, source=str(source), destination=str(destination)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', required=True)
    args = parser.parse_args()
    config, _ = load_config(args.config, args.local)
    relocate(config)
