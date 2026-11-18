import json
import os

import pytest

from tools.reports import observe_context_replica_training as observer
from tools.reports import watch_context_replica_training as watcher


def test_exact_handle_interval_and_terminal(tmp_path):
    launch = tmp_path / 'run-launch.json'
    launch.write_text(json.dumps(dict(container_id='exact', image='image',
                                     manifest_sha256='digest', remote_root='/run')))
    os.utime(launch, (1000, 1000))
    c = dict(operations_root=str(tmp_path), tag='run', manifest_sha256='digest')
    calls = []

    def remote(argv):
        calls.append(argv)
        return json.dumps([dict(Id='exact', Image='image',
                               State=dict(Status='exited', Running=False, ExitCode=1))])

    assert observer.observe(c, now=1899, remote=remote)['status'] == 'not_due_no_remote_call'
    assert not calls
    result = observer.observe(c, now=1900, remote=remote)
    assert watcher.terminal(result)
    assert watcher.summary(result)['exit_code'] == 1
    observer.observe(c, now=2799, remote=remote)
    assert calls == [['docker', 'inspect', 'exact']]


def test_detach_rejects_fast_interval_before_launch(monkeypatch):
    monkeypatch.setattr(watcher, 'load_config', lambda *a: ({'interval_seconds': 60}, {}))
    with pytest.raises(ValueError, match='900-second'):
        watcher.detach('config', 'local')
