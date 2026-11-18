import copy
import pytest
import torch
from multimodal_jev.context_training_checkpoint import capture,restore


def fixture():
    p=torch.nn.Parameter(torch.tensor([.1,.2]));empty=torch.nn.Parameter(torch.empty(0))
    named=[('head',p),('empty',empty)]
    def opt():return torch.optim.AdamW([p,empty],lr=1e-4,foreach=False)
    optimizer=opt();p.square().sum().backward();optimizer.step();optimizer.zero_grad()
    identity=dict(rank=0,world_size=4,completed_updates=1,contract=dict(schedule='fixed',stage='instruction'))
    state=capture(named,optimizer,**identity)
    return named,opt,optimizer,state,identity


def test_resume_exact_next_update_including_unowned_shard():
    named,opt,optimizer,state,identity=fixture();p=named[0][1]
    p.square().sum().backward();optimizer.step();expected=p.detach().clone()
    resumed=opt();restore(named,resumed,state,**identity)
    resumed.zero_grad();p.square().sum().backward();resumed.step()
    assert torch.equal(p,expected)
    for key,fields in optimizer.state_dict()['state'].items():
        for name,v in fields.items():assert torch.equal(v,resumed.state_dict()['state'][key][name])


@pytest.mark.parametrize('fault',['contract','rank','shape','missing','negative','step','fp16'])
def test_reject_before_partial_mutation(fault):
    named,opt,_,state,identity=fixture();state=copy.deepcopy(state)
    if fault=='contract':state['contract']['stage']='task'
    if fault=='rank':state['rank']=1
    if fault=='shape':state['parameters']['empty']=torch.zeros(1)
    if fault=='missing':state['optimizer']['state'].clear()
    if fault=='negative':state['optimizer']['state'][0]['exp_avg_sq'][0]=-1
    if fault=='step':state['optimizer']['state'][0]['step'].fill_(2)
    if fault=='fp16':state['parameters']['head']=state['parameters']['head'].half()
    before=[p.detach().clone() for n,p in named];optimizer=opt()
    with pytest.raises(ValueError):restore(named,optimizer,state,**identity)
    assert all(torch.equal(p,x) for (n,p),x in zip(named,before))
    assert not optimizer.state
