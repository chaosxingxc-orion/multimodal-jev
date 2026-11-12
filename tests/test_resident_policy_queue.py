import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from multimodal_jev.baseline import atomic_json,file_hash
import queue_span_resident_policy_suite as module
import wait_intent_priority_tail as tail


def fixture(tmp_path):
    root=tmp_path/'suite';root.mkdir();(root/'source').mkdir()
    atomic_json(root/'source-manifest.json',{})
    atomic_json(root/'config.json',dict(schema='jev-resident-policy-suite-v1',policies=[dict(key=k) for k in ('initial','d12')],pins={}))
    atomic_json(root/'prepared.json',dict(root=str(root),image='fixed',physical_devices=[4],
        config_sha256=file_hash(root/'config.json'),source_manifest_sha256=file_hash(root/'source-manifest.json')))
    return dict(suite_root=str(root),image='fixed',priority_kind='utility_curves12',pins={str(root/n):file_hash(root/n) for n in ('config.json','prepared.json','source-manifest.json')})


def test_unclaimed_exact_plan_only(tmp_path):
    cfg=fixture(tmp_path);assert module.validate(cfg)[0]==Path(cfg['suite_root'])
    atomic_json(Path(cfg['suite_root'])/'create-attempt.json',{})
    with pytest.raises(ValueError,match='claimed'):module.validate(cfg)


@pytest.mark.parametrize('field,value',[('image','wrong'),('priority_kind','skip')])
def test_wrong_plan_or_dependency_rejected(tmp_path,field,value):
    cfg=fixture(tmp_path);cfg[field]=value
    with pytest.raises(ValueError):module.validate(cfg)


def test_dependency_failure_prevents_launch(tmp_path,monkeypatch):
    cfg=fixture(tmp_path)
    def fail(*a):raise ValueError('utility failed')
    monkeypatch.setattr(tail,'wait_tail',fail)
    monkeypatch.setattr(module,'launch',lambda *a:pytest.fail('launch must not occur'))
    with pytest.raises(ValueError,match='utility failed'):module.run(cfg,tmp_path)


def test_downstream_recomputes_actual_resident_acceptance(tmp_path,monkeypatch):
    cfg=fixture(tmp_path);controller=tmp_path/'controller';(controller/'run').mkdir(parents=True)
    cfg.update(pilot='pilot',manifest_sha256='manifest',priority_predecessor='/utility')
    old=controller/'source/tools/jev/verify_span_resident_policy_suite.py';old.parent.mkdir(parents=True);old.write_text('verifier')
    new=tmp_path/'current-verifier.py';new.write_text('verifier')
    cfg['pins'][str(old)]=file_hash(old);atomic_json(controller/'config.json',cfg)
    atomic_json(controller/'run/terminal.json',dict(exit_code=0));receipt=dict(container_id='exact')
    atomic_json(Path(cfg['suite_root'])/'launch.json',receipt);atomic_json(controller/'run/dispatched.json',receipt)
    audit=dict(passed=True,policies=['initial','d12'],files={str(new):file_hash(new)})
    atomic_json(controller/'run/accepted.json',dict(audit,files={str(old):file_hash(old)}))
    monkeypatch.setattr(module.inspect,'getfile',lambda f:str(new))
    import run_intent_utility_curves as utility
    monkeypatch.setattr(utility,'authenticate_complete',lambda *a:dict(passed=True,pairs=36))
    atomic_json(controller/'run/priority-accepted.json',dict(passed=True,pairs=36))
    calls=[]
    def verify(*args):calls.append(args);return copy.deepcopy(audit)
    monkeypatch.setattr(module,'verify',verify)
    result=tail.authenticate(dict(image='fixed',priority_kind='resident_policy_suite'),controller)
    assert result['policies']==audit['policies'] and calls==[(Path(cfg['suite_root']),'pilot','manifest','fixed')]
    atomic_json(controller/'run/accepted.json',dict(passed=True,policies=['initial'],files={str(old):file_hash(old)}))
    with pytest.raises(ValueError,match='acceptance differs'):module.authenticate_complete(dict(image='fixed'),controller)
    new.write_text('changed verifier')
    with pytest.raises(ValueError,match='identical pinned'):module.authenticate_complete(dict(image='fixed'),controller)
