import json
import os

from tools.reports import watch_context_bucket_training as bucket_watch
from tools.reports import observe_context_macro_training as observer


def test_bucket_observer_uses_exact_bucket_entry_and_shared_interval(tmp_path):
    attempt = tmp_path/'attempt.json'; attempt.write_text('{}'); os.utime(attempt, (1000, 1000))
    (tmp_path/'client-launch.json').write_text(json.dumps(dict(container_id='exact', argv=['bucket-entry'])))
    calls = []
    def remote(argv):
        calls.append(argv)
        if argv[:2] == ['docker', 'inspect']:
            return '[{"Id":"exact","State":{"Status":"running","Running":true}}]'
        return '{"ranks":[],"committed_checkpoints":0}'
    fn = bucket_watch.bind(bucket_watch.observe,
        validate=lambda c: dict(files_sha256='sealed'),
        command=lambda c, role: ['bucket-entry'], inspect_runtime=lambda *a: None)
    c = dict(operations_root=str(tmp_path), remote_root='/run', host='host')
    assert fn(c, now=1899, remote=remote)['status'] == 'not_due_no_remote_call'
    assert not calls
    assert fn(c, now=1900, remote=remote)['status'] == 'observed_exact_attempt'
    assert calls[0] == ['docker', 'inspect', 'exact']
    assert len(calls) == 2
    fn(c, now=2799, remote=remote)
    assert len(calls) == 2
    assert observer.observe.__globals__['command'] is observer.command
    assert bucket_watch.run.__globals__['observe'] is bucket_watch.observe
