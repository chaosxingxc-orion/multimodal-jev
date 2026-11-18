import itertools
import pytest
import torch
from tools.reports.exact_singleton_f1_torch_candidate import expected_reward,loss


def truth(gold,candidates):
    return torch.arange(candidates)[None,:]==torch.tensor(gold)[:,None]


def test_exhaustive_actions_value_and_gradient():
    z=torch.tensor([[.2,-.7,1.1],[-1.,.3,.9]],dtype=torch.float64,requires_grad=True)
    p=z.sigmoid();y=truth([0,2],3);out=expected_reward(p,y,torch.ones_like(y));refs=[]
    for row,gold in enumerate([0,2]):
        ref=p[row].sum()*0
        for action in itertools.product((0,1),repeat=3):
            a=torch.tensor(action,dtype=p.dtype)
            mass=torch.where(a.bool(),p[row],1-p[row]).prod()
            ref=ref+mass*2*action[gold]/(sum(action)+1)
        refs.append(ref)
    ref=torch.stack(refs)
    assert torch.allclose(out,ref,atol=1e-14,rtol=0)
    assert torch.allclose(torch.autograd.grad(out.sum(),z,retain_graph=True)[0],torch.autograd.grad(ref.sum(),z)[0],atol=1e-14,rtol=0)


def test_probability_and_sigmoid_composition_gradcheck():
    p=torch.tensor([[.2,.7,.4],[.8,.1,.3]],dtype=torch.float64,requires_grad=True)
    y=truth([1,0],3);mask=torch.ones_like(y)
    assert torch.autograd.gradcheck(lambda v:expected_reward(v,y,mask),(p,))
    assert torch.autograd.gradcheck(lambda v:expected_reward(v.sigmoid(),y,mask),(p,))


def test_global_normalization_matches_rank_mean_and_microbatch():
    z=torch.linspace(-2,2,24,dtype=torch.float64).reshape(4,6).requires_grad_()
    y=truth([0,2,3,1],6);mask=torch.ones_like(y);real=torch.tensor([True,True,True,False]);weights=torch.tensor([1.,.7,2.,0.],dtype=z.dtype)
    total=loss(z.sigmoid(),y,mask,weights,real,global_real_requests=3)
    g=torch.autograd.grad(total,z)[0];other=z.detach().clone().requires_grad_();parts=[]
    for i in range(4):
        parts.append(loss(other[i:i+1].sigmoid(),y[i:i+1],mask[i:i+1],weights[i:i+1],real[i:i+1],global_real_requests=3,world_size=2)/2)
    got=torch.autograd.grad(sum(parts),other)[0]
    assert torch.allclose(g,got,atol=1e-14,rtol=0)
    assert (got[3]==0).all()


def test_permutation_and_single_candidate_and_endpoints():
    p=torch.tensor([[0.,1.,1.],[1.,0.,0.]],requires_grad=True);y=truth([0,0],3)
    r=expected_reward(p,y,torch.ones_like(y))
    assert torch.equal(r,torch.tensor([0.,1.]))
    assert torch.equal(r,expected_reward(p.flip(1),y.flip(1),torch.ones_like(y)))
    r.sum().backward();assert torch.isfinite(p.grad).all()
    one=torch.tensor([[.7]],requires_grad=True);y=truth([0],1)
    expected_reward(one,y,torch.ones_like(y)).sum().backward();assert one.grad.item()==1


def test_bf16_sigmoid_execution_fp32_probability_gradient():
    z=torch.tensor([[.2,-1.,2.]],dtype=torch.bfloat16,requires_grad=True)
    y=truth([0],3);value=expected_reward(z.float().sigmoid(),y,torch.ones_like(y)).sum()
    assert value.dtype==torch.float32
    value.backward();assert z.grad.dtype==torch.bfloat16 and torch.isfinite(z.grad).all()


@pytest.mark.parametrize('case',['missing','multiple','none','bad_probability','low_precision'])
def test_invalid_truth_and_probabilities(case):
    p=torch.tensor([[.2,.7]]);y=truth([0],2);mask=torch.ones_like(y)
    if case=='missing':mask[0,1]=False
    if case=='multiple':y[:]=True
    if case=='none':y[:]=False
    if case=='bad_probability':p[0,0]=-1
    if case=='low_precision':p=p.bfloat16()
    with pytest.raises(ValueError):expected_reward(p,y,mask)


def test_padding_weight_rejected():
    p=torch.tensor([[.2,.7]]);y=truth([0],2)
    with pytest.raises(ValueError,match='padding'):
        loss(p,y,torch.ones_like(y),torch.ones(1),torch.zeros(1,dtype=torch.bool),global_real_requests=1)
