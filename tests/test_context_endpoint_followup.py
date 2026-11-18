import fcntl
import hashlib
import json

import pytest
from tools.reports.run_context_endpoint_followup import ready,finish_one


def fixture(tmp_path):
    root=tmp_path/'experiment';root.mkdir();(root/'owner.lock').touch()
    (root/'status.json').write_text(json.dumps({'phase':'complete_training_and_selection_curve'}))
    source=tmp_path/'source';source.write_text('fixed')
    pins={str(source):hashlib.sha256(source.read_bytes()).hexdigest()}
    entry=dict(directory=str(root),algorithm='grpo_v1',output=str(tmp_path/'report.json'))
    return root,source,pins,entry


def test_missing_owner_not_created(tmp_path):
    assert ready(tmp_path/'absent') is False
    assert not (tmp_path/'absent').exists()


def test_live_owner_is_not_ready_even_with_completed_status(tmp_path):
    root,_,pins,entry=fixture(tmp_path)
    with (root/'owner.lock').open('r') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not ready(root)
        assert finish_one(entry,'release',pins,compute=lambda *a:pytest.fail('early analysis')) is False


def test_finished_report_releases_predecessor_lock_and_never_overwrites(tmp_path):
    root,_,pins,entry=fixture(tmp_path)
    def compute(entry,release):
        with (root/'owner.lock').open('r') as owner:
            fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return dict(complex_training_admitted=False,scope='test double only')
    assert finish_one(entry,'release',pins,compute=compute)
    with pytest.raises(ValueError,match='existing report'):finish_one(entry,'release',pins,compute=compute)


def test_incomplete_terminal_rejected(tmp_path):
    root,_,_,_=fixture(tmp_path)
    (root/'status.json').write_text(json.dumps({'phase':'stopped_preserve_handles'}))
    with pytest.raises(ValueError,match='without complete curve'):ready(root)


def test_dependencies_rechecked_after_computation(tmp_path):
    _,source,pins,entry=fixture(tmp_path)
    def compute(*args):
        source.write_text('changed');return dict(complex_training_admitted=False)
    with pytest.raises(ValueError,match='dependency changed'):finish_one(entry,'release',pins,compute=compute)
    assert not __import__('pathlib').Path(entry['output']).exists()


def test_compute_failure_never_writes_success(tmp_path):
    _,_,pins,entry=fixture(tmp_path)
    def fail(*args):raise RuntimeError('audit failure')
    with pytest.raises(RuntimeError,match='audit failure'):finish_one(entry,'release',pins,compute=fail)
    assert not __import__('pathlib').Path(entry['output']).exists()
