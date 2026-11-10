import json
import sys
from pathlib import Path

import pytest

from multimodal_jev.baseline import file_hash

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
import watch_omni_selection_to_test as module


def fixture(tmp_path,monkeypatch):
    data={'evaluation-launch.json':{'pid':123},'queue-intent.json':{'order':['parent','p1']},
          'selection-protocol.json':{'prompts':{'parent':'a','p1':'b'}}}
    for name,value in data.items():(tmp_path/name).write_text(json.dumps(value))
    plan=dict(inputs={name:file_hash(tmp_path/name) for name in data},
              selection_pid=123,selection_process_start='birth')
    monkeypatch.setattr(module,'process_start',lambda pid:'birth')
    return plan


def test_waits_on_exact_live_supervisor_and_never_dispatches_partial(tmp_path,monkeypatch):
    plan=fixture(tmp_path,monkeypatch)
    assert module.observe(tmp_path,plan) is False
    monkeypatch.setattr(module,'process_start',lambda pid:None)
    with pytest.raises(RuntimeError,match='missing'):module.observe(tmp_path,plan)
    monkeypatch.setattr(module,'process_start',lambda pid:'reused')
    with pytest.raises(RuntimeError,match='replaced'):module.observe(tmp_path,plan)


def test_requires_complete_success_and_exact_containers(tmp_path,monkeypatch):
    plan=fixture(tmp_path,monkeypatch)
    terminal=tmp_path/'evaluation-terminal.json';terminal.write_text('{"exit_code":1}')
    with pytest.raises(RuntimeError,match='failed'):module.observe(tmp_path,plan)
    terminal.write_text('{"exit_code":0}')
    for key in ['parent','p1']:
        child=tmp_path/key;child.mkdir()
        (child/'launch.json').write_text(json.dumps(dict(container_id=key,image='image')))
    calls=[]
    def inspect(cid):
        calls.append(cid)
        return dict(Id=cid,Image='image',State={'Status':'exited','ExitCode':0})
    monkeypatch.setattr(module,'inspect',inspect)
    assert module.observe(tmp_path,plan) is True and calls==['parent','p1']
    monkeypatch.setattr(module,'inspect',lambda cid:dict(Id=cid,Image='image',State={'Status':'running','ExitCode':0}))
    with pytest.raises(ValueError,match='containers'):module.observe(tmp_path,plan)
    (tmp_path/'selection-protocol.json').write_text('{}')
    with pytest.raises(ValueError,match='identity changed'):module.observe(tmp_path,plan)
