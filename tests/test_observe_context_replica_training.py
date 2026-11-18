import json
from pathlib import Path

import pytest

from tools.reports.observe_context_replica_training import INTERVAL, observe
from tools.reports.watch_context_replica_training import summary, terminal


TAG = 'replica-formal-v1'
CID = 'a' * 64
IMAGE = 'sha256:' + 'b' * 64


def config(tmp_path):
    return dict(operations_root=str(tmp_path), tag=TAG, host='npu',
                manifest_sha256='c' * 64, remote_root='/data/host/run')


def launch(tmp_path, **overrides):
    value = dict(container_id=CID, image=IMAGE, manifest_sha256='c' * 64,
                 remote_root='/data/host/run')
    value.update(overrides)
    (Path(tmp_path) / (TAG + '-launch.json')).write_text(json.dumps(value))


def inspection(*, running=True, status=None):
    return json.dumps([dict(Id=CID, Image=IMAGE,
                            State=dict(Status=status or ('running' if running else 'exited'),
                                       Running=running, ExitCode=0))])


def test_observation_requires_the_exact_launched_handle(tmp_path):
    with pytest.raises(ValueError, match='launched handle'):
        observe(config(tmp_path), now=1.0, remote=lambda args: None)
    launch(tmp_path, manifest_sha256='d' * 64)
    with pytest.raises(ValueError, match='another declared package'):
        observe(config(tmp_path), now=1.0, remote=lambda args: None)


def start(tmp_path):
    return (Path(tmp_path) / (TAG + '-launch.json')).stat().st_mtime


def test_failed_or_early_observation_consumes_the_interval(tmp_path):
    launch(tmp_path)
    calls = []
    due = start(tmp_path) + INTERVAL + 1

    def remote(args):
        calls.append(args)
        if args[0] == 'docker' and args[1] == 'inspect':
            return inspection()
        return json.dumps(dict(progress={'rank-0': dict(phase='update')}, updates={'rank-0': 3},
                               policy_commits=3, last_policy_commit='policy-0003-committed.json',
                               checkpoint_commits=[]))

    first = observe(config(tmp_path), now=due, remote=remote)
    assert first['status'] == 'observed_exact_handle'
    assert first['progress']['updates'] == {'rank-0': 3}
    assert len(calls) == 2
    again = observe(config(tmp_path), now=due + INTERVAL - 1, remote=remote)
    assert again == dict(status='not_due_no_remote_call',
                         next_ordinary_observation_unix=due + INTERVAL)
    assert len(calls) == 2


def test_changed_container_identity_is_refused(tmp_path):
    launch(tmp_path)

    def remote(args):
        return inspection().replace(CID, 'e' * 64)

    with pytest.raises(ValueError, match='identity differs'):
        observe(config(tmp_path), now=start(tmp_path) + INTERVAL + 1, remote=remote)


def test_terminal_state_stops_without_a_success_claim(tmp_path):
    launch(tmp_path)
    result = observe(config(tmp_path), now=start(tmp_path) + INTERVAL + 1,
                     remote=lambda args: inspection(running=False))
    assert result['state']['ExitCode'] == 0
    assert terminal(result)
    row = summary(result)
    assert row['container'] == 'exited' and row['result_present'] is False
    assert not terminal(dict(state=dict(Status='running', Running=True)))
