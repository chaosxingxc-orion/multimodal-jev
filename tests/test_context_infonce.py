import copy
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from multimodal_jev.context_infonce import loss
from multimodal_jev.context_warmup import learning_rate, set_learning_rate
from multimodal_jev.context_training_checkpoint import capture, restore


def compute(z, y, real=None, **kw):
    real = torch.ones(len(z), dtype=torch.bool) if real is None else real
    return loss(z, y, kw.pop('observed', torch.ones_like(y)), real.to(z.dtype), real,
        global_real_requests=kw.pop('global_real_requests', max(1,int(real.sum()))), **kw)[0]


def test_infonce_matches_exclusive_candidate_likelihood_and_gradient():
    z = torch.tensor([[.2,-.3,.7],[-.8,.1,.4]], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[False,False,True],[False,True,False]])
    actual = compute(z,y,temperature=.7)
    expected = F.cross_entropy(z/.7, torch.tensor([2,1]))
    assert torch.allclose(actual,expected,atol=1e-14,rtol=0)
    assert torch.allclose(torch.autograd.grad(actual,z)[0],torch.autograd.grad(expected,z)[0],atol=1e-14,rtol=0)
    assert torch.allclose(compute(z+4,y),compute(z,y))
    assert not torch.allclose(z.sigmoid(),(z+4).sigmoid())


def test_dp8_microbatch_and_all_padding_rank_equal_global_gradient():
    torch.manual_seed(317)
    z = torch.randn(16,3,dtype=torch.float64,requires_grad=True)
    y = F.one_hot(torch.arange(16)%3,3).bool()
    real = torch.arange(16)<9
    full = compute(z,y,real)
    expected = torch.autograd.grad(full,z)[0]
    split = z.detach().clone().requires_grad_()
    partial = sum(compute(split[i:i+1],y[i:i+1],real[i:i+1],
        global_real_requests=9,world_size=8)/8 for i in range(16))
    grad = torch.autograd.grad(partial,split)[0]
    assert torch.allclose(grad,expected,atol=1e-14,rtol=0)
    assert torch.equal(grad[9:],torch.zeros_like(grad[9:]))


@pytest.mark.parametrize('fault',['missing','multiple','absent','temperature'])
def test_reject_unsupported_supervision(fault):
    z=torch.zeros(1,3);y=torch.tensor([[True,False,False]]);kw={}
    if fault=='missing':kw['observed']=torch.tensor([[True,False,True]])
    if fault=='multiple':y[0,1]=True
    if fault=='absent':y[:]=False
    if fault=='temperature':kw['temperature']=0
    with pytest.raises(ValueError):compute(z,y,**kw)


def test_bf16_large_scores_stay_finite():
    z=torch.tensor([[-1000.,1000.]],dtype=torch.bfloat16,requires_grad=True)
    value=compute(z,torch.tensor([[True,False]]));value.backward()
    assert value.dtype==torch.float32 and torch.isfinite(value) and torch.isfinite(z.grad).all()


def test_warmup_checkpoint_resume_exact_next_adam_update():
    spec=dict(kind='linear_warmup_constant',peak_lr=1e-4,warmup_updates=8)
    p=torch.nn.Parameter(torch.tensor([.3,.7]));named=[('head',p)]
    opt=torch.optim.AdamW([p],lr=learning_rate(spec,0,160),foreach=False)
    identity=dict(rank=0,world_size=8,completed_updates=3,contract={'lr_schedule':spec})
    for index in range(3):
        set_learning_rate(opt,spec,index,160);opt.zero_grad();p.square().sum().backward();opt.step()
    state=capture(named,opt,**identity)
    set_learning_rate(opt,spec,3,160);opt.zero_grad();p.square().sum().backward();opt.step()
    expected=p.detach().clone();moments=copy.deepcopy(opt.state_dict())
    resumed=torch.optim.AdamW([p],lr=learning_rate(spec,2,160),foreach=False)
    restore(named,resumed,state,**identity)
    set_learning_rate(resumed,spec,3,160);resumed.zero_grad();p.square().sum().backward();resumed.step()
    assert torch.equal(expected,p)
    for key,fields in moments['state'].items():
        for field,value in fields.items():assert torch.equal(value,resumed.state_dict()['state'][key][field])
    assert learning_rate(spec,0,160)==1e-4/8
    assert learning_rate(spec,7,160)==learning_rate(spec,159,160)==1e-4


def test_declared_dp8_and_dp6_budget_and_trainer_wiring():
    from multimodal_jev.context_topology_batch_plan import derive
    from tools.reports.context_infonce_trainer import topology,generate
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'configs/experiments/context-two-stage-v0.0.4.json').read_text())
    stage1=config['stage1'];c=dict(stage1,physical_devices=list(range(8)),microbatch=8)
    plan=derive(stage1['batch_config'])
    assert (plan['global_batch'],plan['total_updates'],plan['tail_real_queries'])==(512,160,65)
    assert topology(c,8,['davinci'+str(i) for i in range(8)])['local_batch']==64
    assert derive(config['stage2']['batch_config'])['total_updates']==212
    source=generate(root)
    assert 'world_size=4' not in source and 'split(32)' not in source
    assert 'set_learning_rate' in source and 'max(0,start_update-1)' in source
    assert 'probability_coefficient=0.' in source
    compile(source,'<test>','exec')


def test_pinned_actor_reuses_supervision_transport_and_rejects_rl():
    from tools.reports.context_infonce_actor import load
    actor=load(Path(__file__).resolve().parents[1],temperature=1.)
    z=torch.tensor([[.3,-.5]],requires_grad=True);y=torch.tensor([[True,False]])
    options=dict(stage='instruction',ranking_coefficient=1.,probability_coefficient=0.,
        kl_coefficient=0.,global_real_requests=1,world_size=1,
        actions=None,old_joint=None,advantages=None,reference_policy=None)
    value,_=actor.loss(z,y,torch.ones_like(y),torch.ones(1),torch.ones(1,dtype=torch.bool),**options)
    assert torch.allclose(value,compute(z,y))
    with pytest.raises(ValueError):
        actor.loss(z,y,torch.ones_like(y),torch.ones(1),torch.ones(1,dtype=torch.bool),**dict(options,stage='task'))


@pytest.mark.parametrize('extra', [
    {'probability_coefficient': .3},
    {'kl_coefficient': .01},
    {'reference_policy': torch.tensor([[.5, .5]])},
    {'actions': torch.tensor([[True, False]])},
])
def test_stage1_actor_rejects_auxiliary_objectives_and_rl_inputs(extra):
    from tools.reports.context_infonce_actor import load
    actor=load(Path(__file__).resolve().parents[1],temperature=1.)
    z=torch.tensor([[.3,-.5]],requires_grad=True);y=torch.tensor([[True,False]])
    options=dict(stage='instruction',ranking_coefficient=1.,probability_coefficient=0.,
        kl_coefficient=0.,global_real_requests=1,world_size=1,
        actions=None,old_joint=None,advantages=None,reference_policy=None)
    with pytest.raises(ValueError,match='pure supervised InfoNCE'):
        actor.loss(z,y,torch.ones_like(y),torch.ones(1),torch.ones(1,dtype=torch.bool),
                   **dict(options,**extra))
