import copy
import torch
import pytest
from multimodal_jev.intent_resident_replay import compare_state


def state(updates=1):
    p=torch.tensor([.1234567],dtype=torch.float32)
    return dict(rank=0,world_size=4,updates=updates,names=['w'],parameters={'w':p},
        optimizer=dict(param_groups=[dict(params=[0])],state={0:dict(step=torch.tensor(float(updates)),exp_avg=p/10,exp_avg_sq=p/100)}))


@pytest.mark.parametrize('updates',[1,2])
def test_exact_both_resident_boundaries(updates):
    a=state(updates);r=compare_state(a,copy.deepcopy(a),rank=0,updates=updates)
    assert r['exact'] and r['tensors_compared']==4


@pytest.mark.parametrize('field',['master','exp_avg','exp_avg_sq','step','rank','dtype','missing'])
def test_export_equivalence_cannot_hide_master_or_adam_drift(field):
    a=state();b=copy.deepcopy(a)
    if field=='master':
        b['parameters']['w']=torch.nextafter(b['parameters']['w'],torch.tensor([float('inf')]))
        assert torch.equal(a['parameters']['w'].bfloat16(),b['parameters']['w'].bfloat16())
    elif field in ('exp_avg','exp_avg_sq'):b['optimizer']['state'][0][field].add_(1e-7)
    elif field=='step':b['optimizer']['state'][0]['step'].add_(1)
    elif field=='rank':b['rank']=1
    elif field=='dtype':b['parameters']['w']=b['parameters']['w'].bfloat16()
    else:b['optimizer']['state']={}
    with pytest.raises(ValueError):compare_state(a,b,rank=0,updates=1)


@pytest.mark.parametrize('updates',[3,317,634])
def test_continuous_replay_requires_explicit_stage_and_exact_lineage(updates):
    a=state(updates);a.update(stage='continuous_rl_engineering',next_update=updates,
        algorithm='grpo_v1',schedule_sha256='schedule',adapter_id='adapter',reference_policy_id='fixed')
    assert compare_state(a,copy.deepcopy(a),rank=0,updates=updates,continuous=True)['exact']
    with pytest.raises(ValueError):compare_state(a,copy.deepcopy(a),rank=0,updates=updates)
    for key in ('next_update','schedule_sha256','adapter_id','reference_policy_id','algorithm','stage'):
        b=copy.deepcopy(a);b[key]='other'
        with pytest.raises(ValueError):compare_state(a,b,rank=0,updates=updates,continuous=True)
    with pytest.raises(ValueError):compare_state(a,copy.deepcopy(a),rank=0,updates=635,continuous=True)


def test_formal_state_requires_explicit_formal_flag_without_weakening_fixture():
    a=state(64);a.update(stage='simple_stage2_rl',next_update=64,algorithm='grpo_v1',
        schedule_sha256='schedule',adapter_id='rl',reference_policy_id='stage1')
    assert compare_state(a,copy.deepcopy(a),rank=0,updates=64,continuous=True,formal=True)['exact']
    for kwargs in ({},dict(continuous=True),dict(formal=True)):
        with pytest.raises(ValueError):compare_state(a,copy.deepcopy(a),rank=0,updates=64,**kwargs)
