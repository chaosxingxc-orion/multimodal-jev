import copy
import json
from pathlib import Path

import pytest

import tools.jev.prepare_context_rl_evaluation_package as pkg
from tools.jev.launch_context_rl_evaluation import command


@pytest.fixture
def packaged(tmp_path, monkeypatch):
    source = tmp_path / 'original-policy';source.mkdir()
    (source / 'weights.pt').write_bytes(b'fixture-weights')
    evidence = tmp_path / 'commit.json';evidence.write_text('{}')
    model = '/mnt/cephfs/xingchao/models/fixture'
    tasks = '/mnt/cephfs/xingchao/datasets/fixture/tasks.json'
    external = {model + '/model.safetensors': 'weights-digest', tasks: 'tasks-digest'}
    c = dict(source_policy=str(source), source_policy_id='fixture-policy', checkpoint_step=64,
        physical_devices=[7], model=model, tasks=tasks, release='/mnt/cephfs/xingchao/datasets/fixture',
        encoded_audio_cache=pkg.PREFIX + 'context-selection-input-cache-v1', pins=external,
        processor='/stale/processor', reference_branch='/stale/branch',
        rl_evaluation=dict(step=64, run_id='fixture-run', algorithm='grpo_v1',
                          binding_files={str(evidence): pkg.sha(evidence)}))
    spec = {}
    for name in ('template', 'experiment', 'stage1_proof', 'rl_proof'):
        path = tmp_path / (name + '.json');path.write_text('{}');spec[name] = str(path)
    path = tmp_path / 'spec.json';path.write_text(json.dumps(spec))
    # Full-source admission has its own real artifact tests. Here isolate
    # relocation/packaging; no formal policy or remote execution is implied.
    monkeypatch.setattr(pkg, 'prepare', lambda **kw: copy.deepcopy(c))
    original_sha = pkg.sha
    monkeypatch.setattr(pkg, 'sha', lambda p: external[str(p)] if str(p) in external else original_sha(p))
    output = tmp_path / 'package';remote = pkg.PREFIX + 'context-rl-eval-fixture'
    manifest = pkg.package(specification=path, output=output, remote_root=remote)
    return output, remote, manifest


def test_package_relocates_policy_evidence_and_code(packaged):
    root, remote, manifest = packaged
    c = json.loads((root / 'config.json').read_text())
    assert c['source_policy'] == remote + '/policy'
    assert 'processor' not in c and 'reference_branch' not in c
    assert (root / 'policy/weights.pt').read_bytes() == b'fixture-weights'
    assert all(p.startswith(remote + '/evidence/') for p in c['rl_evaluation']['binding_files'])
    assert all(not p.startswith('/stale') for p in c['pins'])
    assert (root / 'source/tools/jev/run_context_rl_selection.py').exists()
    assert pkg.verify(root, pkg.sha(root / 'PACKAGE.json')) == manifest


@pytest.mark.parametrize('fault', ['weight', 'extra', 'manifest', 'symlink'])
def test_package_mutation_rejected(packaged, fault):
    root, _, _ = packaged;digest = pkg.sha(root / 'PACKAGE.json')
    if fault == 'weight':(root / 'policy/weights.pt').write_bytes(b'changed')
    elif fault == 'extra':(root / 'extra').write_text('new')
    elif fault == 'manifest':(root / 'PACKAGE.json').write_text('{}')
    else:(root / 'link').symlink_to(root / 'config.json')
    with pytest.raises(ValueError):pkg.verify(root, digest)


def test_command_isolated_card7_and_read_only_assets(packaged):
    root, remote, m = packaged;c = json.loads((root / 'config.json').read_text())
    argv = command(m, c)
    devices = [argv[i + 1] for i, v in enumerate(argv) if v == '--device']
    assert '/dev/davinci7:/dev/davinci7' in devices
    assert not any('davinci4' in d for d in devices)
    mounts = [argv[i + 1] for i, v in enumerate(argv) if v == '-v']
    for asset in ('source', 'policy', 'evidence', 'config.json'):
        assert remote + '/' + asset + ':' + remote + '/' + asset + ':ro' in mounts
    assert '/mnt/cephfs/xingchao:/mnt/cephfs/xingchao:ro' in mounts
    assert argv[-2:] == ['--output', remote + '/evaluation']
    assert pkg.IMAGE in argv


@pytest.mark.parametrize('field,value', [('physical_devices', [4, 5, 6, 7]), ('encoded_audio_cache', '/mnt/cephfs/cache')])
def test_command_rejects_wrong_device_or_storage(field, value):
    c = dict(physical_devices=[7], encoded_audio_cache=pkg.PREFIX + 'context-selection-input-cache-v1')
    c[field] = value
    with pytest.raises(ValueError):command(dict(remote_root=pkg.PREFIX + 'context-rl-eval-test'), c)


def test_invalid_remote_root_rejected_before_writes(tmp_path):
    with pytest.raises(ValueError):
        pkg.package(specification=tmp_path / 'missing', output=tmp_path / 'package', remote_root='/mnt/cephfs/bad')
    assert not (tmp_path / 'package').exists()


def test_default_launcher_has_no_remote_calls(packaged, monkeypatch, capsys):
    import tools.jev.launch_context_rl_evaluation as launcher
    import sys
    root, _, _ = packaged
    def forbidden(*a, **kw):raise AssertionError('prepare-only attempted remote process')
    monkeypatch.setattr(launcher.subprocess, 'run', forbidden)
    monkeypatch.setattr(launcher.subprocess, 'check_output', forbidden)
    monkeypatch.setattr(sys, 'argv', ['launch', '--package', str(root),
                                    '--manifest-sha256', pkg.sha(root / 'PACKAGE.json')])
    launcher.main()
    assert json.loads(capsys.readouterr().out)['status'] == 'validated_only_no_remote_calls'


def test_nonformal_source_rejection_does_not_create_package(tmp_path, monkeypatch):
    spec = {}
    for name in ('template', 'experiment', 'stage1_proof', 'rl_proof'):
        p = tmp_path / (name + '.json');p.write_text('{}');spec[name] = str(p)
    path = tmp_path / 'spec.json';path.write_text(json.dumps(spec))
    def reject(**kw):raise ValueError('full formal RL run required')
    monkeypatch.setattr(pkg, 'prepare', reject)
    with pytest.raises(ValueError, match='formal RL'):
        pkg.package(specification=path, output=tmp_path / 'package',
                    remote_root=pkg.PREFIX + 'context-rl-eval-rejected')
    assert not (tmp_path / 'package').exists()


def test_launcher_wait_failure_keeps_handle_and_lease(packaged, tmp_path, monkeypatch):
    import fcntl
    import shlex
    import subprocess
    import sys
    import tools.jev.launch_context_rl_evaluation as launcher
    root, remote, _ = packaged
    work = tmp_path / 'launcher-work';work.mkdir();monkeypatch.chdir(work)
    (work / '.local/decision-objective-research').mkdir(parents=True)
    lease = work / '.local/simple-final-v1/npu175-evaluation-training.lock'
    lease.parent.mkdir(parents=True)
    waits, creations, starts = [], [], []
    def require_held():
        with lease.open('a') as stream:
            with pytest.raises(BlockingIOError):fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def output(argv, **kw):
        require_held();cmd = shlex.split(argv[-1])
        if cmd[:3] == ['docker', 'image', 'inspect']:return json.dumps([dict(Id=pkg.IMAGE)])
        if cmd[:2] == ['npu-smi', 'info']:return 'No running processes found in NPU 7'
        if cmd[:2] == ['python3', '-c'] and '.exists()' in cmd[-1]:return 'False\n'
        if cmd[:2] == ['docker', 'create']:creations.append(cmd);return 'exact-container\n'
        return ''
    def process(argv, **kw):
        require_held()
        if argv[0] == 'scp':return subprocess.CompletedProcess(argv, 0)
        cmd = shlex.split(argv[-1])
        launch = work / '.local/decision-objective-research' / (Path(remote).name + '-launch.json')
        assert json.loads(launch.read_text())['container_id'] == 'exact-container'
        if cmd[:2] == ['docker', 'start']:starts.append(cmd);return subprocess.CompletedProcess(argv, 0)
        assert cmd == ['docker', 'wait', 'exact-container']
        waits.append(cmd)
        return subprocess.CompletedProcess(argv, 255 if len(waits) == 1 else 0,
                                           stdout='' if len(waits) == 1 else '0\n')
    monkeypatch.setattr(launcher.subprocess, 'check_output', output)
    monkeypatch.setattr(launcher.subprocess, 'run', process)
    monkeypatch.setattr(launcher.time, 'sleep', lambda seconds: require_held())
    monkeypatch.setattr(sys, 'argv', ['launch', '--package', str(root), '--manifest-sha256',
                                    pkg.sha(root / 'PACKAGE.json'), '--launch'])
    launcher.main()
    assert len(creations) == len(starts) == 1 and len(waits) == 2
    with lease.open('a') as stream:fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
