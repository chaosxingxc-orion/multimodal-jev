import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from prepare_intent_utility_pilot import audit,OBJECTIVES
from run_intent_utility_pilot import await_predecessor
from multimodal_jev.baseline import atomic_json,file_hash


def reference():
    return dict(arm='D',step=0,lr=1e-4,global_requests=8,seed=317,
        initial_adapter='/initial',pins={'before':'hash'})


@pytest.mark.parametrize('objective',OBJECTIVES)
def test_only_objective_changes(objective):
    cfg=reference();candidate=dict(cfg,objective=copy.deepcopy(objective),pins={'after':'hash'})
    audit(cfg,candidate)
    for key,value in [('lr',2e-5),('seed',318),('global_requests',32),('initial_adapter','/different'),('adam_betas',[0,.999])]:
        with pytest.raises(ValueError):audit(cfg,dict(candidate,**{key:value}))
    with pytest.raises(ValueError):audit(cfg,dict(candidate,objective={'name':'uniform-credit-v1','scale':9.}))


def predecessor(tmp_path):
    root=tmp_path/'prior';(root/'run').mkdir(parents=True);out=tmp_path/'out';out.mkdir()
    atomic_json(root/'config.json',{'gate_root':'/gate','pilot':'/pilot','manifest_sha256':'hash'})
    atomic_json(root/'launch.json',{'pid':999999999,'argv':['exact']})
    cfg=dict(predecessor_controller=str(root),predecessor_gate='/gate',image='fixed',
        pins={str(root/n):file_hash(root/n) for n in ('config.json','launch.json')})
    return root,out,cfg


def test_missing_controller_never_dispatches(tmp_path):
    root,out,cfg=predecessor(tmp_path)
    with pytest.raises(ValueError,match='missing'):await_predecessor(cfg,out)


def test_failed_predecessor_stops_chain(tmp_path):
    root,out,cfg=predecessor(tmp_path);atomic_json(root/'run/terminal.json',{'exit_code':1})
    with pytest.raises(ValueError,match='failed'):await_predecessor(cfg,out)


def test_live_predecessor_polls_at_ten_minutes(tmp_path,monkeypatch):
    import os
    import run_intent_utility_pilot as module
    root,out,cfg=predecessor(tmp_path)
    atomic_json(root/'launch.json',dict(pid=os.getpid(),argv=Path(f'/proc/{os.getpid()}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')))
    cfg['pins'][str(root/'launch.json')]=file_hash(root/'launch.json');waits=[]
    def sleep(seconds):
        waits.append(seconds);atomic_json(root/'run/terminal.json',{'exit_code':1})
    monkeypatch.setattr(module.time,'sleep',sleep)
    with pytest.raises(ValueError,match='failed'):await_predecessor(cfg,out)
    assert waits==[600]


def test_success_file_requires_actual_container_completion(tmp_path,monkeypatch):
    import run_intent_utility_pilot as module
    root,out,cfg=predecessor(tmp_path);gate=tmp_path/'gate';gate.mkdir();cfg['predecessor_gate']=str(gate)
    atomic_json(root/'config.json',{'gate_root':str(gate),'pilot':'/pilot','manifest_sha256':'hash'})
    cfg['pins'][str(root/'config.json')]=file_hash(root/'config.json')
    atomic_json(root/'run/terminal.json',{'exit_code':0});launch={'container_id':'cid','image':'fixed'}
    atomic_json(gate/'launch.json',launch);atomic_json(root/'run/dispatched.json',launch)
    monkeypatch.setattr(module,'completed',lambda *a:False)
    with pytest.raises(ValueError,match='actual'):await_predecessor(cfg,out)
    monkeypatch.setattr(module,'completed',lambda *a:True)
    monkeypatch.setattr(module,'verify_gate',lambda *a:{'passed':True})
    await_predecessor(cfg,out);assert (out/'predecessor-accepted.json').exists()
