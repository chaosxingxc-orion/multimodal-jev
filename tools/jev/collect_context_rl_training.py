"""Collect exact terminal training artifacts read-only, including failures.

Transport success does not imply training success. A separate audit checks
population, rewards, policies and optimizer state. No NPU devices are mounted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

from tools.jev.collect_context_rl_selection import check_handle


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def reader_script(remote, mode):
    if mode not in ('manifest', 'archive'):
        raise ValueError('known transport mode required')
    # Includes raw logs and cache receipts needed to investigate failed runs.
    # Sources are immutable in the executed package; rehash them against pins.
    return '''import hashlib,json,sys,tarfile
from pathlib import Path
r=Path(ROOT)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
paths=[]
for name in ('config.json','service-config.json','run.py','PREPARED.json'):
 p=r/name
 if p.is_file():paths.append(p)
for name in ('run','service','supervision','reference-cache'):
 directory=r/name
 if directory.is_symlink():raise ValueError('collection symlink forbidden')
 for p in directory.rglob('*'):
  if p.is_symlink():raise ValueError('collection symlink forbidden')
  if p.is_file():paths.append(p)
paths=sorted(paths)
if any(p.is_symlink() or not p.resolve().is_relative_to(r.resolve()) for p in paths):
 raise ValueError('collection escapes training root')
if MODE=='manifest':
 files={str(p.relative_to(r)):sha(p) for p in paths}
 bad_pins=[];bad_package=[]
 config=json.loads((r/'config.json').read_text())
 manifest=json.loads((r/'PREPARED.json').read_text())
 for name,digest in config['pins'].items():
  p=Path(name)
  if not p.is_file() or sha(p)!=digest:bad_pins.append(name)
 for name,digest in manifest['files'].items():
  relative=Path(name)
  if relative.is_absolute() or '..' in relative.parts:raise ValueError('package path escapes')
  p=r/relative
  if not p.is_file() or sha(p)!=digest:bad_package.append(name)
 print(json.dumps(dict(files=files,bad_pins=bad_pins,bad_package=bad_package)))
else:
 with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
  for p in paths:archive.add(p,arcname=str(p.relative_to(r)),recursive=False)
'''.replace('ROOT', repr(remote)).replace('MODE', repr(mode))


def unpack(archive, output, files):
    output = Path(output)
    with tarfile.open(archive) as tar:
        names = set()
        for member in tar.getmembers():
            relative = Path(member.name)
            if relative.is_absolute() or '..' in relative.parts or not member.isfile():
                raise ValueError('only relative regular artifacts accepted')
            name = str(relative)
            if name in names or name not in files:
                raise ValueError('duplicate or unbound archive artifact')
            names.add(name)
        if names != set(files):raise ValueError('archive population differs')
        tar.extractall(output, filter='data')
    if any(sha(output / name) != digest for name,digest in files.items()):
        raise ValueError('transport hash mismatch; preserve archive')


def collect(launch, output, host='npu'):
    def ssh(argv):return subprocess.check_output(['ssh',host,shlex.join(argv)],text=True,timeout=60)
    observed = json.loads(ssh(['docker','inspect',launch['container_id']]))[0]
    check_handle(launch, observed)
    remote = launch['remote_root']
    reader = ['docker','run','--rm','--network=none','--volumes-from',launch['container_id']+':ro',
              '--entrypoint','python',launch['image']]
    # Hashing full training output may take minutes: no artificial 60s cutoff.
    wire = json.loads(subprocess.check_output(['ssh',host,shlex.join(reader+['-c',reader_script(remote,'manifest')])],text=True))
    output = Path(output);output.mkdir(exist_ok=False)
    receipt = dict(status='transport_pending_not_audited',container_id=launch['container_id'],
        image=launch['image'],remote_root=remote,state=observed['State'],**wire)
    path = output.with_suffix('.transport.json')
    with path.open('x') as stream:json.dump(receipt,stream,indent=2)
    archive = output.with_suffix('.tar')
    with archive.open('xb') as stream:
        subprocess.run(['ssh',host,shlex.join(reader+['-c',reader_script(remote,'archive')])],stdout=stream,check=True)
    unpack(archive,output,wire['files'])
    bindings = {'config.json':'config_sha256','service-config.json':'service_config_sha256','PREPARED.json':'manifest_sha256'}
    mismatches = [name for name,key in bindings.items() if wire['files'].get(name) != launch[key]]
    receipt.update(archive_sha256=sha(archive),launch_binding_mismatches=mismatches,
                   status='transport_complete_independent_audit_required')
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    # Preserve failed execution output and all mismatch evidence before refusal.
    if mismatches or wire['bad_pins'] or wire['bad_package']:
        raise ValueError('executed package/config changed; collection preserved')
    return receipt


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--host',default='npu')
    a=p.parse_args();result=collect(json.loads(a.launch.read_text()),a.output,a.host)
    print(json.dumps(dict(status=result['status'],container_id=result['container_id'],
                          state=result['state'],files=len(result['files'])),indent=2))
