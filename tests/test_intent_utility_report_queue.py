import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import queue_intent_utility_report as module
from multimodal_jev.baseline import atomic_json,file_hash


def fixture(tmp_path):
    protocol=tmp_path/'protocol.json';atomic_json(protocol,dict(arms=[{'name':a} for a in ('c','u','s')]))
    prior=tmp_path/'controller';prior.mkdir();atomic_json(prior/'config.json',dict(plan_root=str(tmp_path),image='fixed'))
    out=tmp_path/'run';out.mkdir()
    return dict(protocol=str(protocol),protocol_sha256=file_hash(protocol),image='fixed',
        priority_predecessor=str(prior),priority_kind='utility_curves12',
        pins={str(p):file_hash(p) for p in (protocol,prior/'config.json')}),out


def test_predecessor_failure_prevents_report(tmp_path,monkeypatch):
    cfg,out=fixture(tmp_path)
    def fail(*args):raise ValueError('actual predecessor failed')
    monkeypatch.setattr(module,'wait_tail',fail)
    monkeypatch.setattr(module,'report',lambda *a:pytest.fail('scored before acceptance'))
    with pytest.raises(ValueError,match='predecessor failed'):module.run(cfg,out)
    assert not (out/'report.json').exists()


@pytest.mark.parametrize('bad',[None,'missing_arm','missing_update','protocol','scope'])
def test_full_registered_curves_only(tmp_path,monkeypatch,bad):
    cfg,out=fixture(tmp_path);monkeypatch.setattr(module,'wait_tail',lambda *a:None)
    r=dict(complete=True,role='learning',updates=12,protocol_sha256=cfg['protocol_sha256'],
        curves={a:[{'update':i} for i in range(1,13)] for a in ('c','u','s')})
    if bad=='missing_arm':del r['curves']['u']
    elif bad=='missing_update':r['curves']['u'].pop()
    elif bad=='protocol':r['protocol_sha256']='other'
    elif bad=='scope':r['role']='test'
    monkeypatch.setattr(module,'report',lambda *a:r)
    if bad:
        with pytest.raises(ValueError,match='complete registered'):module.run(cfg,out)
        assert not (out/'report.json').exists()
    else:
        module.run(cfg,out);assert module.read(out/'report.json')==r


def test_wrong_predecessor_study_rejected_before_wait(tmp_path,monkeypatch):
    cfg,out=fixture(tmp_path);p=Path(cfg['priority_predecessor'])/'config.json'
    atomic_json(p,dict(plan_root='/other',image='fixed'));cfg['pins'][str(p)]=file_hash(p)
    monkeypatch.setattr(module,'wait_tail',lambda *a:pytest.fail('wrong study'))
    with pytest.raises(ValueError,match='another registered'):module.run(cfg,out)
