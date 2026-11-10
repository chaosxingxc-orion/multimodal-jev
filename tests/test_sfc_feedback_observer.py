import json
from pathlib import Path
import subprocess
import sys
import pytest
from multimodal_jev.baseline import atomic_json,file_hash
from test_sfc_jev import load_tool

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))


def fixture(tmp_path):
    root=tmp_path/'observer';root.mkdir();partition=tmp_path/'partition.json';partition.write_text('{}')
    plan=dict(schema='sfc-jev-feedback-observer-v1',API_dispatch=False,observation_seconds=110000,
        automatic_retry=False,source_files={},partition=str(partition),partition_sha256=file_hash(partition),
        dependency=dict(root='/exact-train78'))
    atomic_json(root/'observer-plan.json',plan)
    return load_tool('watch_sfc_teacher_feedback'),root,plan


def test_wait_then_prepare_then_downstream_contract_without_api(tmp_path,monkeypatch):
    m,root,plan=fixture(tmp_path);events=[];ready=iter([False,True])
    monkeypatch.setattr(m,'observe_dependency',lambda _:next(ready))
    monkeypatch.setattr(m.time,'sleep',lambda _:events.append('wait'))
    def prepare(probe,partition,pin,out):
        assert str(probe)=='/exact-train78' and pin==plan['partition_sha256']
        events.append('prepare');out.mkdir();atomic_json(out/'prepared.json',{'API_sent':False});return {'API_sent':False}
    monkeypatch.setattr(m,'prepare',prepare)
    monkeypatch.setattr(m,'prepared_feedback',lambda *a:events.append('validate'))
    m.execute(root)
    assert events==['wait','prepare','validate']
    assert json.loads((root/'result.json').read_text())['API_sent'] is False
    assert not list(root.rglob('*executor*'))
    with pytest.raises(FileExistsError):m.execute(root)


def test_transient_observation_retries_same_dependency_only(tmp_path,monkeypatch):
    m,root,plan=fixture(tmp_path);seen=[]
    def observe(dep):
        seen.append(dep)
        if len(seen)==1:raise subprocess.TimeoutExpired('ssh',20)
        raise RuntimeError('failed predecessor')
    monkeypatch.setattr(m,'observe_dependency',observe);monkeypatch.setattr(m.time,'sleep',lambda _:None)
    monkeypatch.setattr(m,'prepare',lambda *a:pytest.fail('cannot prepare'))
    with pytest.raises(RuntimeError,match='predecessor'):m.execute(root)
    assert seen==[plan['dependency']]*2 and not (root/'preparation-intent.json').exists()


@pytest.mark.parametrize('stage',['prepare','validate'])
def test_failed_preparation_keeps_intent_and_no_replay(tmp_path,monkeypatch,stage):
    m,root,_=fixture(tmp_path);monkeypatch.setattr(m,'observe_dependency',lambda _:True)
    def prepare(*args):
        args[-1].mkdir();atomic_json(args[-1]/'prepared.json',{})
        if stage=='prepare':raise ValueError('preparation failure')
        return {}
    def validate(*args):raise ValueError('downstream failure')
    monkeypatch.setattr(m,'prepare',prepare);monkeypatch.setattr(m,'prepared_feedback',validate)
    with pytest.raises(ValueError):m.execute(root)
    assert (root/'preparation-intent.json').exists() and not (root/'result.json').exists()
    with pytest.raises(FileExistsError):m.execute(root)


def test_changed_partition_prevents_observation(tmp_path,monkeypatch):
    m,root,plan=fixture(tmp_path);Path(plan['partition']).write_text('changed')
    monkeypatch.setattr(m,'observe_dependency',lambda _:pytest.fail('changed partition'))
    with pytest.raises(ValueError,match='partition'):m.execute(root)
