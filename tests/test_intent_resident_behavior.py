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
from multimodal_jev.intent_resume import authenticate as authenticate_resume
from test_intent_resume import fixture as state_fixture


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


def continuation_fixture(tmp_path,monkeypatch):
    behavior,_,audit=fixture(tmp_path,monkeypatch)
    state=tmp_path/'state';state.mkdir();cfg=state_fixture(state)
    previous_path=Path(cfg['previous_root'])/'config.json'
    previous=verifier.read(previous_path)
    for key in ('model_path','pilot','manifest_sha256','runtime_profile'):
        previous[key]=cfg[key]=behavior[key]
    atomic_json(previous_path,previous)
    gate_path=Path(cfg['previous_suite'])/'config.json';gate=verifier.read(gate_path)
    gate.update(model=cfg['model_path'],runtime_profile=cfg['runtime_profile'],base_id='base')
    atomic_json(gate_path,gate)
    cfg['pins'][str(previous_path)]=file_hash(previous_path)
    cfg['pins'][str(gate_path)]=file_hash(gate_path)
    original=copy.deepcopy(cfg)
    cfg.update(resident_behavior=behavior['resident_behavior'],acceptance_root=behavior['acceptance_root'])
    cfg['pins'].update(behavior['pins'])
    return cfg,original,audit


def test_resident_behavior_preserves_authenticated_optimizer_lineage(tmp_path,monkeypatch):
    cfg,_,_=continuation_fixture(tmp_path,monkeypatch)
    accepted=authenticate_resume(cfg)
    assert accepted['previous_updates']==1
    assert accepted['resident_behavior']['policy_id']=='base:adapter'
    shard=Path(cfg['previous_root'])/'run/rank-3/optimizer-local.pt'
    shard.write_text('corrupt state')
    with pytest.raises(ValueError,match='state changed'):authenticate_resume(cfg)


def test_resident_policy_must_match_state_policy_even_if_each_receipt_valid(tmp_path,monkeypatch):
    cfg,_,_=continuation_fixture(tmp_path,monkeypatch)
    gate=Path(cfg['previous_suite'])/'config.json';data=verifier.read(gate)
    data['base_id']='different-base';atomic_json(gate,data);cfg['pins'][str(gate)]=file_hash(gate)
    with pytest.raises(ValueError,match='selected resident policy'):authenticate_resume(cfg)


def test_binder_authenticates_both_state_and_new_behavior(tmp_path,monkeypatch):
    cfg,original,audit=continuation_fixture(tmp_path,monkeypatch)
    path=tmp_path/'continuation.json';atomic_json(path,original)
    monkeypatch.setattr(binder,'checkpoint_identity',lambda p:('base' if p=='base' else 'adapter',{}))
    bound,proof=binder.bind(path,audit,'first')
    assert bound['previous_suite']==original['previous_suite']
    assert bound['previous_root']==original['previous_root']
    assert bound['acceptance_root']==cfg['acceptance_root']
    assert proof['resume_state']['previous_updates']==1
    failed=Path(original['previous_suite'])/'controller-terminal.json'
    atomic_json(failed,{'exit_code':1});original['pins'][str(failed)]=file_hash(failed)
    atomic_json(path,original)
    with pytest.raises(ValueError,match='acceptance failed'):binder.bind(path,audit,'first')
