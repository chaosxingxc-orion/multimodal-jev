import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.intent_resident_behavior import authenticate
from test_span_resident_suite_host import full_fixture
import verify_span_resident_policy_suite as verifier
import bind_intent_resident_behavior as binder


def fixture(tmp_path,monkeypatch):
    root=full_fixture(tmp_path,monkeypatch)
    audit=verifier.verify(root,'pilot','manifest','fixed');path=tmp_path/'audit.json';atomic_json(path,audit)
    cfg=dict(model_path='base',initial_adapter='adapter',runtime_profile='profile',pilot='pilot',manifest_sha256='manifest',
        acceptance_root=str(root/'run/first'),pins=dict(audit['files']),
        resident_behavior=dict(schema='jev-resident-behavior-v1',audit=str(path),key='first'))
    cfg['pins'][str(path)]=file_hash(path)
    return cfg,audit,path


def test_complete_actor_side_binding(tmp_path,monkeypatch):
    cfg,_,_=fixture(tmp_path,monkeypatch)
    assert authenticate(cfg,'base:adapter')['key']=='first'


@pytest.mark.parametrize('bad',['missing_pin','policy','key','population','gate','launch','partial_audit'])
def test_rejects_stale_or_incomplete_binding(tmp_path,monkeypatch,bad):
    cfg,audit,path=fixture(tmp_path,monkeypatch)
    if bad=='missing_pin':cfg['pins'].pop(next(iter(audit['files'])))
    elif bad=='policy':
        with pytest.raises(ValueError):authenticate(cfg,'base:other')
        return
    elif bad=='key':cfg['resident_behavior']['key']='unknown'
    elif bad=='population':cfg['manifest_sha256']='different'
    elif bad=='gate':cfg['acceptance_root']+='/different'
    elif bad=='launch':
        p=Path(audit['root'])/'launch.json';r=verifier.read(p);r['container_id']='changed';atomic_json(p,r)
    else:
        audit['files'].pop(str(Path(cfg['acceptance_root'])/'run/hf/result.json'))
        atomic_json(path,audit);cfg['pins'][str(path)]=file_hash(path)
    with pytest.raises(ValueError):authenticate(cfg,'base:adapter')


def test_binder_reverifies_host_before_input_overlay(tmp_path,monkeypatch):
    cfg,audit,path=fixture(tmp_path,monkeypatch);original=dict(cfg)
    original.pop('resident_behavior');original['acceptance_root']='old-gate';original['pins']={}
    training=tmp_path/'training.json';atomic_json(training,original)
    monkeypatch.setattr(binder,'checkpoint_identity',lambda p:(p,{}))
    new,evidence=binder.bind(training,path,'first')
    assert new['acceptance_root']==cfg['acceptance_root'] and evidence['policy_id']=='base:adapter'
    assert original['acceptance_root']=='old-gate'
    monkeypatch.setattr(verifier,'completed',lambda *a:False)
    with pytest.raises(ValueError,match='successful resident'):binder.bind(training,path,'first')
