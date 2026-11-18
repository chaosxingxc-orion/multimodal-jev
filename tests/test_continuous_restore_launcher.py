"""Exercise the actual recovery launcher against an isolated SSH/Docker host."""
import hashlib
import json
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
import tarfile

import pytest

from multimodal_jev.resident_service_admission import IMAGE


@pytest.fixture
def host(tmp_path, monkeypatch):
    repository = tmp_path / 'repo'
    target = repository / 'tools/jev/launch_simple_continuous_restore.py'
    target.parent.mkdir(parents=True)
    source = Path(__file__).parents[1] / 'tools/jev/launch_simple_continuous_restore.py'
    target.write_text(source.read_text())
    (repository / 'src').mkdir()
    (repository / 'src/example.py').write_text('VALUE = 1\n')
    local = tmp_path / 'state'
    local.mkdir()
    base = '/data/xingchao/multimodal-jev/simple-final-v1'
    stem = 'sampled-npu-grpo_v1'
    train = base + '/' + stem + '-resident-train-v1'
    audit = base + '/' + stem + '-resident-train-audit-v1'
    restore = base + '/' + stem + '-resident-restore-v1'
    files, calls, copies = {}, [], []
    mode = dict(occupied=False, root_exists=False)

    def put(path, value):
        path.write_text(json.dumps(value) + '\n')

    for role, root in [('train', train), ('train-audit', audit)]:
        tag = stem + '-resident-' + role
        put(local / (tag + '-launch.json'), dict(container_id=tag, image=IMAGE, remote_root=root))
        put(local / (tag + '-observed.json'), dict(container_id=tag, image=IMAGE,
            state=dict(Status='exited', ExitCode=0)))
    accepted = dict(passed=True, updates=3, ranks=4, exact_master_export=True,
                    exact_peft_reload=True, exact_pre_update_recovery=True)
    verified = local / (stem + '-resident-train-audit-verified.json')
    put(verified, accepted)
    files[audit + '/run/state-verified.json'] = verified.read_bytes()
    files[audit + '/training-observed.json'] = (local / (stem + '-resident-train-observed.json')).read_bytes()
    files[train + '/config.json'] = json.dumps(dict(physical_devices=[0, 1, 2, 3],
        pins={train + '/source/old.py': 'old', '/data/immutable-input': 'input'},
        global_batch=128)).encode()
    files[train + '/run/checkpoint-0003-committed.json'] = json.dumps(dict(reference_policy_id='fixed-reference')).encode()
    put(local / 'prefix-prepared.json', dict(argv=['docker', 'create']))

    def remote(argv, **kwargs):
        assert argv[:2] == ['ssh', 'npu']
        command = shlex.split(argv[2])
        calls.append(command)
        if command[:2] == ['python3', '-c']:
            return str(mode['root_exists']) + '\n'
        if command[0] == 'cat':
            return files[command[1]].decode()
        if command[0] == 'sha256sum':
            return hashlib.sha256(files[command[1]]).hexdigest() + ' ' + command[1] + '\n'
        if command == ['npu-smi', 'info']:
            return '\n'.join('No running processes found in NPU ' + str(i)
                             for i in range(3 if mode['occupied'] else 4))
        if command[0] in ('mkdir', 'tar'):
            return ''
        if command[:2] == ['docker', 'create']:
            return 'recovery-container\n'
        if command[:2] == ['docker', 'start']:
            return 'recovery-container\n'
        if command[:2] == ['docker', 'inspect']:
            return json.dumps([dict(State=dict(Status='running', Running=True))])
        raise AssertionError(command)

    def copy(argv, **kwargs):
        assert argv[0] == 'scp' and argv[2].startswith('npu:')
        copies.append(argv)
        files[argv[2][4:]] = Path(argv[1]).read_bytes()

    monkeypatch.setattr(subprocess, 'check_output', remote)
    monkeypatch.setattr(subprocess, 'run', copy)
    monkeypatch.setattr(sys, 'argv', [str(target), '--state-dir', str(local)])
    monkeypatch.chdir(local)
    return dict(target=target, local=local, files=files, calls=calls, copies=copies,
                mode=mode, stem=stem, train=train, audit=audit, restore=restore)


def launch(host):
    runpy.run_path(str(host['target']), run_name='__main__')


@pytest.mark.parametrize('failure', ['handle', 'root', 'busy', 'failed_source', 'unaccepted', 'hash'])
def test_recovery_rejects_invalid_admission_without_mutating_host(host, failure):
    local, stem = host['local'], host['stem']
    if failure == 'handle':
        (local / (stem + '-resident-restore-created.json')).write_text('{}')
    elif failure == 'root':
        host['mode']['root_exists'] = True
    elif failure == 'busy':
        host['mode']['occupied'] = True
    elif failure == 'failed_source':
        path = local / (stem + '-resident-train-observed.json')
        value = json.loads(path.read_text()); value['state']['ExitCode'] = 1
        path.write_text(json.dumps(value))
    elif failure == 'unaccepted':
        path = local / (stem + '-resident-train-audit-verified.json')
        value = json.loads(path.read_text()); value['exact_peft_reload'] = False
        path.write_text(json.dumps(value))
    else:
        host['files'][host['audit'] + '/run/state-verified.json'] += b' '
    with pytest.raises((ValueError, RuntimeError)):
        launch(host)
    assert not host['copies']
    assert not any(c[0] in ('mkdir', 'tar', 'docker') for c in host['calls'])


def test_recovery_preserves_audited_path_and_pins_fresh_source(host):
    launch(host)
    root, local, stem = host['restore'], host['local'], host['stem']
    config = json.loads(host['files'][root + '/config.json'])
    resume = config['continuous_resume']
    assert resume['training_observed'] == host['audit'] + '/training-observed.json'
    assert resume['completed_updates'] == 3 and resume['reference_policy_id'] == 'fixed-reference'
    for key in ('audit', 'audit_launch', 'audit_observed', 'training_observed'):
        path = resume[key]
        assert config['pins'][path] == hashlib.sha256(host['files'][path]).hexdigest()
    assert host['train'] + '/source/old.py' not in config['pins']
    assert config['pins']['/data/immutable-input'] == 'input'
    with tarfile.open(local / (stem + '-resident-restore-source.tar.gz')) as archive:
        assert 'src/example.py' in archive.getnames()
        for item in archive.getmembers():
            assert config['pins'][root + '/source/' + item.name] == hashlib.sha256(archive.extractfile(item).read()).hexdigest()
    command = next(c for c in host['calls'] if c[:2] == ['docker', 'create'])
    devices = [command[i + 1] for i, v in enumerate(command) if v == '--device']
    assert devices[:4] == ['/dev/davinci%d:/dev/davinci%d' % (i, i) for i in range(4)]
    mounts = [command[i + 1] for i, v in enumerate(command) if v == '-v']
    assert '/mnt/cephfs/xingchao:/mnt/cephfs/xingchao:ro' in mounts
    assert root + '/source:' + root + '/source:ro' in mounts
    receipt = json.loads((local / (stem + '-resident-restore-launch.json')).read_text())
    assert receipt['next_regular_observation_not_before'] - receipt['created_unix'] >= 600
    assert receipt['config_sha256'] == hashlib.sha256(host['files'][root + '/config.json']).hexdigest()
    assert sum(c[:2] == ['docker', 'start'] for c in host['calls']) == 1
