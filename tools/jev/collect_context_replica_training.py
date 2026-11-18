"""Collect exact terminal artifacts for any declared replica training layout.

The transport set is read from the launch record's package layout, so one
collector serves every topology. Transport success still implies nothing about
training success; a separate audit is required.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from tools.jev.collect_context_rl_selection import check_handle
from tools.jev.collect_context_rl_training import sha, unpack


def transport_names(launch):
    layout = launch['layout']
    names = list(layout['configs'])
    names += [layout['services'][name] for name in sorted(layout['services'])]
    return names, list(layout['directories']), layout


def bindings(launch, layout):
    expected = {'config.json': launch['config_sha256'],
                'PREPARED.json': launch['manifest_sha256']}
    for name, path in layout['services'].items():
        expected[path] = launch['services'][name]
    return expected


def reader_script(remote, mode, names, directories):
    if mode not in ('manifest', 'archive'):
        raise ValueError('known transport mode required')
    return '''import hashlib,json,sys,tarfile
from pathlib import Path
r=Path(ROOT)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
paths=[]
for name in NAMES:
 p=r/name
 if p.is_file():paths.append(p)
for name in DIRECTORIES:
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
'''.replace('ROOT', repr(remote)).replace('MODE', repr(mode)).replace(
        'NAMES', repr(names)).replace('DIRECTORIES', repr(directories))


def collect(launch, output, host='npu'):
    def ssh(argv):
        return subprocess.check_output(['ssh', host, shlex.join(argv)], text=True, timeout=60)
    names, directories, layout = transport_names(launch)
    observed = json.loads(ssh(['docker', 'inspect', launch['container_id']]))[0]
    check_handle(launch, observed)
    remote = launch['remote_root']
    reader = ['docker', 'run', '--rm', '--network=none', '--volumes-from',
              launch['container_id'] + ':ro', '--entrypoint', 'python', launch['image']]
    wire = json.loads(subprocess.check_output(
        ['ssh', host, shlex.join(reader + ['-c', reader_script(remote, 'manifest', names, directories)])],
        text=True))
    output = Path(output); output.mkdir(exist_ok=False)
    receipt = dict(status='transport_pending_not_audited', container_id=launch['container_id'],
                   image=launch['image'], remote_root=remote, state=observed['State'], **wire)
    path = output.with_suffix('.transport.json')
    with path.open('x') as stream:
        json.dump(receipt, stream, indent=2)
    archive = output.with_suffix('.tar')
    with archive.open('xb') as stream:
        subprocess.run(['ssh', host, shlex.join(
            reader + ['-c', reader_script(remote, 'archive', names, directories)])],
            stdout=stream, check=True)
    unpack(archive, output, wire['files'])
    mismatches = [name for name, digest in bindings(launch, layout).items()
                  if wire['files'].get(name) != digest]
    receipt.update(archive_sha256=sha(archive), launch_binding_mismatches=mismatches,
                   status='transport_complete_independent_audit_required')
    path.write_text(json.dumps(receipt, indent=2) + '\n')
    if mismatches or wire['bad_pins'] or wire['bad_package']:
        raise ValueError('executed package/config changed; collection preserved')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--host', default='npu')
    args = parser.parse_args()
    result = collect(json.loads(args.launch.read_text()), args.output, args.host)
    print(json.dumps(dict(status=result['status'], container_id=result['container_id'],
                          state=result['state'], files=len(result['files'])), indent=2))
