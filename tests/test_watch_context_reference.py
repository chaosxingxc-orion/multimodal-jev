import json

import pytest

from tools.reports import watch_context_reference as watcher


def test_watcher_waits_for_complete_launch_and_uses_original_wait(tmp_path, monkeypatch):
    monkeypatch.setattr(watcher, 'commit_json', lambda p, data: p.write_text(json.dumps(data)))
    base, output = tmp_path/'ops', tmp_path/'watch'
    (base/'finalization').mkdir(parents=True)
    output.mkdir()
    layout = dict(operations_root=str(base), package=str(tmp_path/'package'), host='npu', tag='reference')
    remote_root, cid = '/srv/reference', 'a'*64
    config = dict(layout, manifest_sha256='b'*64, reference_job={'remote_root':remote_root})
    (base/'finalization/config.json').write_text(json.dumps(config))
    launch = base/'reference-launch.json'
    launch.write_text('{')
    assert watcher.discover(layout, remote_root, output, None) is None

    record = dict(host='npu', remote_root=remote_root, image='sha256:'+'c'*64,
                  container_id=cid, manifest_sha256=config['manifest_sha256'])
    launch.write_text(json.dumps(record))
    binding = watcher.discover(layout, remote_root, output, None)
    assert binding['container_id'] == cid
    assert json.loads((output/'binding.json').read_text()) == binding
    monkeypatch.setattr(watcher, 'observe', lambda _: pytest.fail('must not make a remote call'))
    assert watcher.current_status(layout, output, binding, observe_due=False) == (None, None)

    wait = base/'reference-wait.json'
    wait.write_text(json.dumps(dict(container_id='d'*64, exit_code=1)))
    with pytest.raises(ValueError, match='different job'):
        watcher.current_status(layout, output, binding, observe_due=True)
    wait.write_text(json.dumps(dict(container_id=cid, exit_code=1)))
    result, terminal = watcher.current_status(layout, output, binding, observe_due=True)
    assert result is None
    assert terminal == dict(status='original_scoring_container_exited', container_id=cid, exit_code=1)

    launch.write_text(json.dumps(dict(record, container_id='d'*64)))
    with pytest.raises(ValueError, match='launch changed'):
        watcher.discover(layout, remote_root, output, binding)
