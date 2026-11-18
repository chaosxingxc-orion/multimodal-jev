#!/usr/bin/env python3
"""Collect one exact terminal evaluation container, including failed output.

Read-only remote transport, not an evaluation success certificate. Audit the
result with audit_context_rl_selection.py after successful terminal collection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile


def check_handle(launch, observed):
    if observed['Id'] != launch['container_id'] or observed['Image'] != launch['image']:
        raise ValueError('container or image identity differs')
    if observed['State']['Status'] != 'exited':
        raise ValueError('not terminal; retain the same handle, do not restart')


def unpack(archive, output, files):
    with tarfile.open(archive) as tar:
        actual = set()
        for member in tar.getmembers():
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts or not member.isfile():
                raise ValueError('only named regular collection files are accepted')
            name = str(path)
            if name in actual or name not in files:
                raise ValueError('duplicate or unbound archive file')
            actual.add(name)
        if actual != set(files):
            raise ValueError('archive population differs from transport receipt')
        tar.extractall(output, filter='data')
    for name, digest in files.items():
        if hashlib.sha256((Path(output) / name).read_bytes()).hexdigest() != digest:
            raise ValueError('transport artifact hash mismatch: ' + name)


def collect(launch, output, host='npu'):
    def ssh(argv):
        return subprocess.check_output(['ssh', host, shlex.join(argv)], text=True, timeout=60)
    observed = json.loads(ssh(['docker', 'inspect', launch['container_id']]))[0]
    check_handle(launch, observed)
    remote = launch['remote_root']
    reader = ['docker', 'run', '--rm', '--network=none', '--volumes-from',
              launch['container_id'] + ':ro', '--entrypoint', 'python', launch['image']]
    script = '''import hashlib,json,sys,tarfile
from pathlib import Path
r=Path(ROOT)
paths=sorted([r/'config.json']+[p for p in (r/'evaluation').rglob('*') if p.is_file()])
if any(p.is_symlink() or not p.resolve().is_relative_to(r.resolve()) for p in paths):
 raise ValueError('collection symlink forbidden')
files={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
if MODE=='manifest':print(json.dumps(files))
else:
 with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as t:
  for p in paths:t.add(p,arcname=str(p.relative_to(r)),recursive=False)
'''.replace('ROOT', repr(remote))
    files = json.loads(ssh(reader + ['-c', script.replace('MODE', repr('manifest'))]))
    output = Path(output)
    output.mkdir(exist_ok=False)
    receipt = dict(container_id=launch['container_id'], image=launch['image'], state=observed['State'],
                   remote_root=remote, files=files, status='transport_pending_not_audited')
    receipt_path = output.with_suffix('.transport.json')
    with receipt_path.open('x') as stream:
        json.dump(receipt, stream, indent=2)
    archive = output.with_suffix('.tar')
    with archive.open('xb') as stream:
        remote_command = reader + ['-c', script.replace('MODE', repr('archive'))]
        subprocess.run(['ssh', host, shlex.join(remote_command)],
                       stdout=stream, check=True)
    unpack(archive, output, files)
    if files['config.json'] != launch['config_sha256']:
        raise ValueError('config differs from launched identity; collection preserved')
    receipt.update(status='transport_complete_independent_audit_required',
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--launch', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--host', default='npu')
    a = p.parse_args()
    receipt = collect(json.loads(a.launch.read_text()), a.output, a.host)
    print(json.dumps(dict(container_id=receipt['container_id'], state=receipt['state'],
                          files=len(receipt['files']), status=receipt['status']), indent=2))
