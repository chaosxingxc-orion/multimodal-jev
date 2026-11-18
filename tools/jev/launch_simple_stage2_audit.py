#!/usr/bin/env python3
"""Launch an independent CPU audit of an explicitly stopped formal segment.

Prefix audits certify a requested recovery point, never select one implicitly.
The exact source container must be exited; timeouts never trigger a restart.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import time

repository = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repository / 'src'))
from multimodal_jev.resident_service_admission import IMAGE
from multimodal_jev.simple_stage2_audit import audit_boundary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--training-tag', required=True)
    p.add_argument('--audit-tag', required=True)
    p.add_argument('--completed-updates', type=int, help='Explicit stopped prefix; omit for terminal634')
    p.add_argument('--state-dir', type=Path, default=repository / '.local/simple-final-v1')
    a = p.parse_args()
    if any(not re.fullmatch(r'stage2-[a-z0-9_-]+', t) for t in (a.training_tag, a.audit_tag)):
        p.error('explicit stage2 tag with lowercase letters, digits, underscores or hyphens required')
    local = a.state_dir; base = '/data/xingchao/multimodal-jev/simple-final-v1'
    remote = base + '/' + a.audit_tag + '-v1'
    def ssh(argv):
        return subprocess.check_output(['ssh', 'npu', shlex.join(argv)], text=True, timeout=60)
    def read(path): return json.loads(Path(path).read_text())
    def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def put(path, value): Path(path).write_text(json.dumps(value, indent=2) + '\n')
    def copy(source, name):
        subprocess.run(['scp', str(source), 'npu:' + remote + '/' + name], check=True,
                       capture_output=True, timeout=60)
    if any((local / (a.audit_tag + suffix)).exists() for suffix in ('-created.json', '-launch.json')):
        raise RuntimeError('reconcile existing exact audit handle; no redispatch')
    config = dict(training_observed=remote + '/training-observed.json', pins={})
    if a.completed_updates is not None:
        config.update(audit_mode='stopped_prefix', completed_updates=a.completed_updates)
    updates, prefix = audit_boundary(config)
    launch = read(local / (a.training_tag + '-launch.json'))
    if launch['image'] != IMAGE:
        raise ValueError('fixed source image required')
    training = launch['remote_root']
    if not training.startswith(base + '/') or Path(training).parent != Path(base):
        raise ValueError('source must be a direct child of local runtime root')
    # Live process state, not a terminal marker or a stale observation file.
    raw = json.loads(ssh(['docker', 'inspect', launch['container_id']]))[0]
    if (raw['Id'] != launch['container_id'] or raw['Image'] != IMAGE
            or raw['State']['Status'] != 'exited' or type(raw['State'].get('ExitCode')) is not int
            or (not prefix and raw['State']['ExitCode'] != 0)):
        raise ValueError('exact exited source required; terminal audit also requires success')
    observed = dict(container_id=raw['Id'], image=raw['Image'], state=raw['State'],
                    observed_unix=time.time(), raw_inspect=raw)
    train = json.loads(ssh(['cat', training + '/config.json']))
    if 'stage2_initialization' not in train or 'sampled_probe' in train:
        raise ValueError('formal Stage2 source required')
    if ssh(['python3', '-c', 'from pathlib import Path;print(Path(' + repr(remote) + ').exists())']).strip() == 'True':
        raise RuntimeError('existing audit root; no overwrite')
    config['training_root'] = training
    for name in ('config.json', 'launch.json'):
        config['pins'][training + '/' + name] = ssh(['sha256sum', training + '/' + name]).split()[0]
    observed_path = local / (a.audit_tag + '-training-observed.json'); put(observed_path, observed)
    config['pins'][config['training_observed']] = digest(observed_path)
    archive_path = local / (a.audit_tag + '-source.tar.gz')
    with tarfile.open(archive_path, 'w:gz') as archive:
        for directory in ('src', 'tests', 'tools/jev'):
            for path in sorted((repository / directory).rglob('*.py')):
                relative = str(path.relative_to(repository)); archive.add(path, arcname=relative)
                config['pins'][remote + '/source/' + relative] = digest(path)
    config_path = local / (a.audit_tag + '-config.json'); put(config_path, config)
    commands = [
        ['python', '-m', 'pytest', '-q', '-p', 'no:cacheprovider'] + [remote + '/source/tests/' + n for n in
            ('test_simple_stage2_audit.py', 'test_simple_stage2_resume.py', 'test_simple_resident_train_audit.py')],
        ['python', remote + '/source/tools/jev/verify_simple_resident_train.py',
         '--config', remote + '/config.json', '--output', remote + '/run']]
    runner = '''import json,subprocess,time
from pathlib import Path
root=Path(__file__).parent
try:
 for i,command in enumerate(COMMANDS):
  with (root/('phase-'+str(i)+'.log')).open('w') as out:
   code=subprocess.call(command,stdout=out,stderr=subprocess.STDOUT)
  if code:raise RuntimeError('phase '+str(i)+' exit '+str(code))
 (root/'terminal.json').write_text(json.dumps(dict(exit_code=0,finished_unix=time.time())))
except BaseException as exc:
 (root/'terminal.json').write_text(json.dumps(dict(exit_code=1,error=repr(exc),finished_unix=time.time())))
 raise
'''.replace('COMMANDS', repr(commands))
    run_path = local / (a.audit_tag + '-run.py'); run_path.write_text(runner)
    ssh(['mkdir', '-p', remote + '/source', remote + '/cache', remote + '/home'])
    for source, name in ((archive_path, 'source.tar.gz'), (config_path, 'config.json'),
                         (observed_path, 'training-observed.json'), (run_path, 'run.py')):
        copy(source, name)
    ssh(['tar', '-xzf', remote + '/source.tar.gz', '-C', remote + '/source'])
    old = read(local / 'prefix-prepared.json')['argv']
    historical = [old[i+1] for i, x in enumerate(old) if x == '-v'
                  and ':/home/xingchao/workspace/multimodal-jev/' in old[i+1]]
    argv = ['docker', 'create', '--pull=never', '--network=none', '--name', 'multimodal-jev-' + a.audit_tag,
            '--cpus=6', '--memory=24g', '--log-driver=json-file', '--log-opt', 'max-size=100m', '--log-opt', 'max-file=3']
    for mount in [base + ':' + base + ':ro', remote + ':' + remote, remote + '/source:' + remote + '/source:ro',
                  remote + '/home:/root', '/mnt/cephfs/xingchao:/mnt/cephfs/xingchao:ro',
                  '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro'] + historical:
        argv += ['-v', mount]
    env = dict(PYTHONPATH=remote + '/source/src', PYTHONDONTWRITEBYTECODE='1', TORCH_DEVICE_BACKEND_AUTOLOAD='0',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    for key in ('TMPDIR', 'HF_HOME', 'XDG_CACHE_HOME', 'TRANSFORMERS_CACHE', 'TORCHINDUCTOR_CACHE_DIR',
                'TRITON_CACHE_DIR', 'ASCEND_WORK_PATH', 'ASCEND_CACHE_PATH'):
        env[key] = remote + '/cache'
    for key, value in env.items(): argv += ['-e', key + '=' + value]
    argv += [IMAGE, 'python', remote + '/run.py']
    put(local / (a.audit_tag + '-prepared.json'), dict(argv=argv))
    cid = ssh(argv).strip(); put(local / (a.audit_tag + '-created.json'), dict(container_id=cid))
    now = time.time()
    receipt = dict(container_id=cid, image=IMAGE, host='175', remote_root=remote,
        created_unix=now, next_regular_observation_not_before=now+600, config_sha256=digest(config_path),
        scope='Independent formal Stage2 ' + ('stopped prefix' if prefix else 'terminal') + ' audit',
        training_tag=a.training_tag, completed_updates=updates)
    put(local / (a.audit_tag + '-launch.json'), receipt)
    copy(local / (a.audit_tag + '-launch.json'), 'launch.json')
    ssh(['docker', 'start', cid])
    receipt['state'] = json.loads(ssh(['docker', 'inspect', cid]))[0]['State']
    put(local / (a.audit_tag + '-launch.json'), receipt)
    print(cid, receipt['state']['Status'])


if __name__ == '__main__': main()
