import json
from pathlib import Path
import subprocess
import sys

from multimodal_jev.baseline import file_hash
from tools.reports import observe_context_reference as observer
from tools.reports.observe_context_replica_training import INTERVAL


def test_exact_reference_handle_counts_only_committed_distinct_rows(tmp_path, monkeypatch):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return file_hash(path)

    base, package, output = (tmp_path/k for k in ('ops', 'package', 'observations'))
    remote_root = tmp_path/'remote'
    journal = remote_root/'reference-journal/try-1'
    build_hash = write(package/'reference-build-config.json', dict(
        macro_client=dict(journal_root=str(journal.parent), attempt_id=journal.name, max_queries=4),
        expected_population=10, source_policy_id='endpoint'))
    image = 'sha256:' + 'b'*64
    manifest = write(package/'PREPARED.json', dict(image=image, remote_root=str(remote_root),
        files={'reference-build-config.json':build_hash}))
    binding = dict(operations_root=str(base), package=str(package), manifest_sha256=manifest,
                   host='worker', tag='reference', reference_job={'remote_root':str(remote_root)})
    config = base/'finalization/config.json'
    c = dict(operations=dict(path=str(config), sha256=write(config, binding)), observations_root=str(output))
    launch = base/'reference-launch.json'
    cid = 'a'*64
    write(launch, dict(host='worker', image=image, manifest_sha256=manifest, container_id=cid,
                       remote_root=str(remote_root)))
    monkeypatch.setattr(observer, 'check_paths', lambda _:None)
    calls = []

    def remote(args):
        calls.append(args[:3])
        assert args[2] == cid
        if args[1] == 'inspect':
            return json.dumps([dict(Id=cid, Image=image, State={'Running':True})])
        return subprocess.check_output([sys.executable, *args[4:]], text=True)

    def commit(index, count):
        name = f'batch-{index:08d}'
        write(journal/name/'COMPLETE.json', dict(batch_id=name, policy_id='endpoint',
            completed_indices=list(range(count)), status='complete_macro_inference', actor_update_performed=False))

    commit(0, 4)
    (journal/'batch-00000001').mkdir()
    now = launch.stat().st_mtime + INTERVAL + 1
    first = observer.run(c, now=now, remote=remote)
    assert first['progress'] == dict(committed_distinct_inputs=4, configured_source_recordings=10,
        committed_batches=1, policy_id='endpoint', first_batch_committed=True, independent_audit_complete=False)
    assert observer.run(c, now=now+1, remote=remote)['status'] == 'not_due_no_remote_call'
    assert len(calls) == 2

    commit(1, 3)
    tail = observer.run(c, now=now+INTERVAL, remote=remote)
    assert tail['progress']['committed_distinct_inputs'] == 7
    assert tail['progress']['independent_audit_complete'] is False
