from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'tools/jev'))
import watch_omni_compact_verl as queue


def state(status='running', code=0):
    return dict(Id=queue.CONTAINER, Image=queue.IMAGE, State=dict(Status=status, ExitCode=code, OOMKilled=False))


@pytest.mark.parametrize('status', ['created', 'running', 'paused', 'restarting'])
def test_live_states_are_waits(status):
    assert queue.candidate_ready(state(status)) is False


@pytest.mark.parametrize('change', ['id', 'image', 'oom', 'dead', 'failed', 'unknown'])
def test_bad_terminal_never_admits(change):
    item = state('exited')
    if change == 'id': item['Id'] = 'other'
    if change == 'image': item['Image'] = 'other'
    if change == 'oom': item['State']['OOMKilled'] = True
    if change == 'dead': item['State']['Status'] = 'dead'
    if change == 'failed': item['State']['ExitCode'] = 1
    if change == 'unknown': item['State']['Status'] = 'unknown'
    with pytest.raises((ValueError, RuntimeError)): queue.candidate_ready(item)


def test_successful_terminal_is_ready():
    assert queue.candidate_ready(state('exited')) is True


@pytest.mark.parametrize('failure', [False, True])
def test_transient_observation_then_exactly_one_launch(tmp_path, monkeypatch, failure):
    plan = dict(dependency_seconds=14400, launcher_seconds=900, python='/usr/bin/python3', cwd=str(tmp_path))
    monkeypatch.setattr(queue, 'check_plan', lambda *a: deepcopy(plan))
    observations = iter([subprocess.TimeoutExpired('ssh', 45), state(), state('exited')])
    def inspect():
        value = next(observations)
        if isinstance(value, Exception): raise value
        return value
    monkeypatch.setattr(queue, 'inspect_candidate', inspect)
    monkeypatch.setattr(queue.time, 'sleep', lambda _: None)
    calls = []
    def launch(argv, **kwargs):
        assert (tmp_path/'dispatch-intent.json').exists()
        calls.append(argv)
        return subprocess.CompletedProcess(argv, int(failure))
    monkeypatch.setattr(queue.subprocess, 'run', launch)
    if failure:
        with pytest.raises(RuntimeError, match='no replay'): queue.execute(tmp_path, 'pin')
    else:
        queue.execute(tmp_path, 'pin')
        assert json.loads((tmp_path/'progress.json').read_text())['phase'] == 'actor_dispatched'
    assert len(calls) == 1
    with pytest.raises(FileExistsError): queue.execute(tmp_path, 'pin')
    assert len(calls) == 1


def test_candidate_failure_prevents_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, 'check_plan', lambda *a: dict(dependency_seconds=14400))
    monkeypatch.setattr(queue, 'inspect_candidate', lambda: state('exited', 1))
    with pytest.raises(RuntimeError): queue.execute(tmp_path, 'pin')
    assert not (tmp_path/'dispatch-intent.json').exists()


@pytest.mark.parametrize('change', [None, 'plan', 'source', 'launcher', 'retry', 'budget', 'python'])
def test_plan_binds_immutable_dispatch(tmp_path, change):
    (tmp_path/'source').mkdir(); source = tmp_path/'source/test.py'; source.write_text('# fixture\n')
    launcher = tmp_path/'launcher.py'; launcher.write_text('# fixture launcher\n')
    plan = dict(schema='omni-compact-verl-dispatch-v1', candidate_id=queue.CONTAINER, image=queue.IMAGE,
                automatic_retry=False, dependency_seconds=14400, launcher_seconds=900,
                python='/usr/bin/python3', cwd='/home/xingchao/workspace/multimodal-jev',
                launcher_sha256=queue.file_hash(launcher), source_files={'test.py':queue.file_hash(source)})
    if change == 'retry': plan['automatic_retry'] = True
    if change == 'budget': plan['dependency_seconds'] = 99999
    if change == 'python': plan['python'] = 'arbitrary'
    path = tmp_path/'plan.json'; path.write_text(json.dumps(plan)); pin = queue.file_hash(path)
    if change == 'plan': path.write_text('{}')
    if change == 'source': source.write_text('# changed')
    if change == 'launcher': launcher.write_text('# changed')
    if change is None: assert queue.check_plan(tmp_path, pin) == plan
    else:
        with pytest.raises(ValueError): queue.check_plan(tmp_path, pin)
