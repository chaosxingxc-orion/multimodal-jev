import copy
import pytest
import torch
from multimodal_jev.intent_residency_comparison import compare_states


def state():
    model=torch.nn.Linear(3,1,bias=False)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0.,foreach=False,fused=False)
    for _ in range(2):
        opt.zero_grad();model(torch.ones(1,3)).square().sum().backward();opt.step()
    return dict(rank=0,world_size=4,updates=2,names=['weight'],
                parameters={'weight':model.weight.detach().clone()},optimizer=copy.deepcopy(opt.state_dict()))


def test_exact_actual_adam_state_compares():
    s=state();r=compare_states(s,copy.deepcopy(s))
    assert r['passed'] and r['tensors_compared']==4 and r['elements']==10


@pytest.mark.parametrize('field',['master','exp_avg','exp_avg_sq','step'])
def test_sub_bf16_master_and_each_optimizer_change_rejected(field):
    s=state();t=copy.deepcopy(s)
    value=t['parameters']['weight'] if field=='master' else t['optimizer']['state'][0][field]
    value.add_(1e-7 if field!='step' else 1.)
    with pytest.raises(ValueError,match='tensor differs'):compare_states(s,t)


@pytest.mark.parametrize('field,value',[('rank',1),('updates',3),('world_size',1),('names',['other'])])
def test_wrong_topology_rejected(field,value):
    s=state();t=copy.deepcopy(s);t[field]=value
    with pytest.raises(ValueError):compare_states(s,t)


def test_optimizer_change_and_nonfinite_rejected():
    s=state();t=copy.deepcopy(s);t['optimizer']['param_groups'][0]['lr']=2e-4
    with pytest.raises(ValueError):compare_states(s,t)
    t=copy.deepcopy(s);t['parameters']['weight'].fill_(float('nan'))
    with pytest.raises(ValueError):compare_states(s,t)
