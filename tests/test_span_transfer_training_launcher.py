"""Regression coverage for stale-snapshot and physical-device launch failures."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from multimodal_jev.baseline import file_hash
from test_span_transfer_runtime import fixture as transfer_fixture


@pytest.fixture
def launcher_environment(tmp_path, monkeypatch):
    module_path = Path(__file__).resolve().parents[1] / 'tools/jev/launch_span_transfer_training.py'
    spec = importlib.util.spec_from_file_location('training_launcher', module_path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    protocol, _ = transfer_fixture(tmp_path)
    repo = tmp_path / 'repository'
    files = {
        'src/package/nested/runtime.py': 'DEVICE_SOURCE = "current"\n',
        'tools/jev/run_span_transfer_chain.py': 'CONTROLLER = "current"\n',
        'tools/jev/check_span_rule_runtime.py': 'CHECKER = "current"\n',
        'configs/tasks/slurp/temporal/tasks-v1.json': '{}\n',
        'docs/engineering/span-rule-transfer-protocol-v1.json': json.dumps(protocol),
    }
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    parent = tmp_path / 'experiments'
    parent.mkdir()
    monkeypatch.setattr(launcher, '__file__', str(repo / 'tools/jev/launch_span_transfer_training.py'))
    monkeypatch.setattr(launcher, 'Path', lambda value: parent if str(value) ==
                        '/mnt/cephfs/xingchao/experiments/multimodal-jev' else Path(value))
    monkeypatch.setattr(launcher.subprocess, 'check_output', lambda *a, **kw: 'reviewed-commit\n')
    calls = []

    def launch(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout='container-id\n', stderr='',
                               check_returncode=lambda: None)

    monkeypatch.setattr(launcher.subprocess, 'run', launch)
    return launcher, repo, parent, calls


@pytest.mark.parametrize('device', [4, 7])
def test_complete_current_snapshot_and_explicit_device(launcher_environment, device):
    launcher, repo, parent, calls = launcher_environment
    root = parent / 'new-branch'
    args = SimpleNamespace(root=root, seed=29, method='conditional', arm='structured',
                           auxiliary_weight=.2, device=device, launch=True)
    receipt = launcher.prepare(args)
    manifest = json.loads((root / 'source-manifest.json').read_text())
    config = json.loads((root / 'config.json').read_text())
    assert config['physical_devices'] == [device]
    assert config['seed'] == 29 and config['num_updates'] == 12
    assert set(manifest['files']) == {
        'src/package/nested/runtime.py', 'tools/jev/run_span_transfer_chain.py',
        'tools/jev/check_span_rule_runtime.py', 'configs/tasks/slurp/temporal/tasks-v1.json',
    }
    for name, digest in manifest['files'].items():
        assert digest == file_hash(repo / name) == file_hash(root / 'source' / name)
        assert config['pins'][str(root / 'source' / name)] == digest
    argv = receipt['argv']
    mapped_devices = [argv[i + 1] for i, value in enumerate(argv) if value == '--device']
    assert [d for d in mapped_devices if '/dev/davinci' in d and 'manager' not in d] == [
        f'/dev/davinci{device}:/dev/davinci{device}']
    assert str(root / 'source/tools/jev/run_span_transfer_chain.py') in argv
    assert receipt['container_id'] == 'container-id' and len(calls) == 1
    assert json.loads((repo / 'docs/engineering/new-branch-launch.json').read_text()) == receipt
    with pytest.raises(ValueError, match='exclusive'):
        launcher.prepare(args)
    assert len(calls) == 1


def test_invalid_branch_is_rejected_before_snapshot_or_launch(launcher_environment):
    launcher, _, parent, calls = launcher_environment
    root = parent / 'invalid-branch'
    args = SimpleNamespace(root=root, seed=29, method='conditional', arm='final_only',
                           auxiliary_weight=.2, device=4, launch=True)
    with pytest.raises(ValueError, match='branch differs'):
        launcher.prepare(args)
    assert not root.exists() and not calls
