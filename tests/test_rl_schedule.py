import importlib.util
import json
from pathlib import Path
import pytest


def module():
    path=Path(__file__).parents[1]/'tools/jev/run_audio_rl.py'
    spec=importlib.util.spec_from_file_location('schedule',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def test_stage_completion_requires_terminal_and_result(tmp_path):
    m=module()
    assert not m.completed(tmp_path)
    (tmp_path/'terminal.json').write_text('{"exit_code":0}')
    with pytest.raises(RuntimeError,match='without result'):m.completed(tmp_path)
    (tmp_path/'result.json').write_text('{}')
    assert m.completed(tmp_path)
    (tmp_path/'terminal.json').write_text('{"exit_code":1}')
    with pytest.raises(RuntimeError,match='failed stage'):m.completed(tmp_path)


def test_existing_incomplete_stage_is_never_restarted(tmp_path,monkeypatch):
    m=module();stage=tmp_path/'actor';stage.mkdir()
    calls=[]
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:calls.append(a))
    with pytest.raises(RuntimeError,match='process inspection'):m.execute('update_audio_actor.py',{'x':1},stage)
    assert not calls


def test_completed_stage_does_not_rerun_or_accept_new_config(tmp_path,monkeypatch):
    m=module();stage=tmp_path/'rollout';stage.mkdir()
    (stage/'terminal.json').write_text('{"exit_code":0}')
    (stage/'result.json').write_text('{}')
    (tmp_path/'rollout-config.json').write_text('{"seed":17}')
    calls=[];monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:calls.append(a))
    m.execute('sample_audio_policy.py',{'seed':17},stage)
    assert not calls
    with pytest.raises(ValueError,match='config changed'):m.execute('sample_audio_policy.py',{'seed':29},stage)
