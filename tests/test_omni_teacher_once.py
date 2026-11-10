import importlib.util
import json
import sys
from pathlib import Path

import pytest

from multimodal_jev.baseline import file_hash

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
spec=importlib.util.spec_from_file_location('teacher_once',Path(__file__).parents[1]/'tools/jev/run_omni_teacher_once.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_failed_teacher_process_is_terminal_and_not_replayed(tmp_path,monkeypatch):
    executor=tmp_path/'executor.py';executor.write_text('raise SystemExit(1)')
    cfg=tmp_path/'executor-config.private.json';cfg.write_text('{}')
    monkeypatch.setattr(module,'EXECUTOR_SHA',file_hash(executor))
    (tmp_path/'teacher-call-intent.json').write_text(json.dumps(dict(config_sha256=file_hash(cfg))))
    with pytest.raises(RuntimeError,match='no retry'):
        module.execute_once(tmp_path,executor)
    assert json.loads((tmp_path/'process-terminal.json').read_text())['exit_code']==1
    with pytest.raises(FileExistsError):
        module.execute_once(tmp_path,executor)
    assert not (tmp_path/'candidates').exists()


def test_control_failure_prevents_feedback_and_api(tmp_path,monkeypatch):
    controls=tmp_path/'controls';(controls/'actual').mkdir(parents=True)
    identity=dict(container_id='cid',image='image')
    (controls/'actual/launch.json').write_text(json.dumps(identity))
    executor=tmp_path/'executor.py';executor.write_text('raise SystemExit(1)')
    reference=tmp_path/'reference.json';reference.write_text('{}')
    monkeypatch.setattr(module,'EXECUTOR_SHA',file_hash(executor))
    (tmp_path/'queue-intent.json').write_text(json.dumps(dict(controls=str(controls),
        controls_identity={'actual':identity},executor=str(executor),reference_config=str(reference),
        reference_config_sha256=file_hash(reference))))
    monkeypatch.setattr(module,'wait',lambda *args:{'Status':'exited','ExitCode':1})
    with pytest.raises(RuntimeError,match='no teacher dispatch'):
        module.run(tmp_path)
    assert not (tmp_path/'request').exists()
