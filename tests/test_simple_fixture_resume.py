import copy
import pytest
torch = pytest.importorskip('torch')
from multimodal_jev.simple_fixture_resume import continuous_plan


def fixture():
    parameter=torch.nn.Parameter(torch.tensor([.5]))
    optimizer=torch.optim.AdamW([parameter],lr=1e-4)
    for _ in range(3):
        parameter.sum().backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    config=dict(sampled_probe=dict(fixture_initialization='restore_continuous_then_one_update',
        algorithm='grpo_v1',stop_update=4),continuous_resume=dict(algorithm='grpo_v1',reference_policy_id='base:fixed'))
    recovered=dict(stage='continuous_rl_engineering',completed_updates=3,
        policy_id='base:third',adapter_id='third',reference_policy_id='base:fixed')
    return parameter,optimizer,config,recovered


def test_resume_plan_preserves_adam_and_next_update():
    parameter,optimizer,config,recovered=fixture()
    before=copy.deepcopy(optimizer.state_dict())
    result=continuous_plan(config,recovered,optimizer,base_id='base',adapter_id='third')
    assert result==dict(policy_id='base:third',reference_policy_id='base:fixed',start_update=3,stop_update=4)
    for key,value in before['state'][0].items():assert torch.equal(value,optimizer.state_dict()['state'][0][key])
    parameter.sum().backward();optimizer.step();assert optimizer.state[parameter]['step'].item()==4


@pytest.mark.parametrize('fault',['formal','mode','stop','cursor','bool_cursor','stage','policy','adapter',
    'reference','algorithm','empty_adam','adam_step','stale_gradient'])
def test_wrong_scope_or_reset_state_rejected(fault):
    parameter,optimizer,config,recovered=fixture()
    if fault=='formal':config['stage2_initialization']={}
    elif fault=='mode':config['sampled_probe']['fixture_initialization']='other'
    elif fault=='stop':config['sampled_probe']['stop_update']=5
    elif fault in ('cursor','bool_cursor'):recovered['completed_updates']=2 if fault=='cursor' else True
    elif fault=='stage':recovered['stage']='simple_stage2_rl'
    elif fault=='policy':recovered['policy_id']='base:other'
    elif fault=='adapter':recovered['adapter_id']='other'
    elif fault=='reference':recovered['reference_policy_id']='moving'
    elif fault=='algorithm':config['sampled_probe']['algorithm']='rloo_v1'
    elif fault=='empty_adam':optimizer.state.clear()
    elif fault=='adam_step':optimizer.state[parameter]['step'].fill_(0)
    else:parameter.grad=torch.ones_like(parameter)
    with pytest.raises(ValueError):continuous_plan(config,recovered,optimizer,base_id='base',adapter_id='third')


def test_entry_uses_shared_restore_and_continuous_callback(tmp_path,monkeypatch):
    import importlib.util
    import sys
    import types
    from pathlib import Path
    path=Path(__file__).parents[1]/'tools/jev/probe_simple_continuous_resume_update.py'
    spec=importlib.util.spec_from_file_location('fixture_resume_entry',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    _,_,config,_=fixture();config.update(global_batch=128,microbatch_per_rank=1)
    config['continuous_resume']['completed_updates']=3
    monkeypatch.setenv('WORLD_SIZE','4');monkeypatch.setenv('LOCAL_RANK','0')
    import multimodal_jev.local_runtime_storage as storage
    monkeypatch.setattr(storage,'require_training_storage',lambda p:None)
    callback=object();seen=[]
    def restore(c,root,rank,*,after_restore):
        assert after_restore is callback and rank==0 and c is config
        seen.append('shared_restore_with_continuous')
    monkeypatch.setitem(sys.modules,'probe_simple_continuous_restore',types.SimpleNamespace(run=restore))
    monkeypatch.setitem(sys.modules,'probe_simple_resident_train',types.SimpleNamespace(continuous=callback))
    module.run(config,tmp_path/'run',0)
    assert seen==['shared_restore_with_continuous']
