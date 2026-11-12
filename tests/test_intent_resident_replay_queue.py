import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import queue_intent_resident_replay as tool
from multimodal_jev.baseline import atomic_json,file_hash


def test_predecessor_failure_cannot_dispatch(tmp_path,monkeypatch):
    p=tmp_path/'prior';(p/'run').mkdir(parents=True)
    atomic_json(p/'config.json',{});atomic_json(p/'launch.json',{'pid':123,'argv':['python']});atomic_json(p/'run/terminal.json',{'exit_code':1})
    cfg=dict(predecessor=str(p),pins={str(p/n):file_hash(p/n) for n in ('config.json','launch.json')})
    monkeypatch.setattr(tool,'completed',lambda *a:pytest.fail('failed predecessor reached Docker'))
    with pytest.raises(ValueError,match='predecessor failed'):tool.await_predecessor(cfg,tmp_path)


@pytest.mark.parametrize('bad',['hash','rankcount','requests','exit'])
def test_artifact_success_requires_complete_independent_evidence(tmp_path,bad):
    report=dict(passed=True,updates=2,ranks=[{}]*4,full_gate={'requests':178},adapter_id='a',files={})
    if bad=='rankcount':report['ranks'].pop()
    if bad=='requests':report['full_gate']['requests']=1
    atomic_json(tmp_path/'verified.json',report)
    atomic_json(tmp_path/'ACCEPTED.json',dict(passed=True,adapter_id='a',verified_sha256='bad' if bad=='hash' else file_hash(tmp_path/'verified.json')))
    atomic_json(tmp_path/'controller-terminal.json',dict(exit_code=1 if bad=='exit' else 0))
    with pytest.raises(ValueError):tool.verify_accepted(tmp_path)
