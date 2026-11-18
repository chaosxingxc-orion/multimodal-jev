import pytest
import torch
from torch.nn import functional as F
from multimodal_jev.context_candidate_objective import loss


def compute(z,y,mask=None,weights=None,real=None,**kw):
    return loss(z,y,torch.ones_like(y) if mask is None else mask,
        z.new_ones(z.shape[0]) if weights is None else weights,
        torch.ones(z.shape[0],dtype=torch.bool) if real is None else real,
        global_real_requests=kw.pop('global_real_requests',z.shape[0]),
        ranking_coefficient=kw.pop('ranking_coefficient',1.),
        probability_coefficient=kw.pop('probability_coefficient',.2),**kw)[0]


def test_independent_pair_reference_and_finite_difference():
    z=torch.tensor([[.2,-.3,.7,-.5]],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([[True,False,True,False]])
    actual=compute(z,y,temperature=.7,margin=.1)
    ref=sum(F.softplus((z[0,n]-z[0,p]+.1)/.7) for p in [0,2] for n in [1,3])/4
    ref=ref+.2*F.binary_cross_entropy_with_logits(z,y.double())
    assert torch.allclose(actual,ref,atol=1e-14,rtol=0)
    grad=torch.autograd.grad(actual,z)[0]
    for j in range(4):
        a=z.detach().clone();b=a.clone();a[0,j]+=1e-6;b[0,j]-=1e-6
        fd=(compute(a,y,temperature=.7,margin=.1)-compute(b,y,temperature=.7,margin=.1))/2e-6
        assert abs(fd-grad[0,j])<1e-9


def test_missing_labels_have_no_gradient_and_no_fabricated_negatives():
    z=torch.tensor([[.2,99.,-.5],[.8,-.1,3.]],requires_grad=True)
    y=torch.tensor([[True,False,False],[False,True,False]])
    mask=torch.tensor([[True,False,True],[False,False,False]])
    value=compute(z,y,mask);value.backward()
    assert z.grad[0,1]==0 and torch.equal(z.grad[1],torch.zeros(3))
    assert torch.allclose(value,compute(z[:1].detach()[:,[0,2]],y[:1,[0,2]])/2)


@pytest.mark.parametrize('positive',[True,False])
def test_single_polarity_retains_probability_supervision_without_pairs(positive):
    z=torch.tensor([[.2,-.3]],requires_grad=True);y=torch.full((1,2),positive)
    ranked=compute(z,y,probability_coefficient=0)
    assert ranked==0
    value=compute(z,y,ranking_coefficient=0,probability_coefficient=1)
    value.backward();assert (z.grad<0).all() if positive else (z.grad>0).all()


def test_global_denominator_microbatch_and_rank_mean_gradient():
    z=torch.randn(4,3,dtype=torch.float64,requires_grad=True)
    y=torch.tensor([[True,False,False],[True,True,False],[False,False,False],[True,False,False]])
    real=torch.tensor([True,True,True,False]);weights=torch.tensor([1.,2.,.7,0.],dtype=torch.float64)
    full=compute(z,y,real=real,weights=weights,global_real_requests=3)
    grad=torch.autograd.grad(full,z)[0]
    split=z.detach().clone().requires_grad_();total=split.sum()*0
    for i in range(4):
        total=total+compute(split[i:i+1],y[i:i+1],real=real[i:i+1],weights=weights[i:i+1],
            global_real_requests=3,world_size=2)/2
    assert torch.allclose(torch.autograd.grad(total,split)[0],grad,atol=1e-14,rtol=0)


def test_ranking_is_offset_invariant_but_probability_anchor_is_not():
    z=torch.tensor([[.2,-.3,.1]]);y=torch.tensor([[True,False,True]])
    assert torch.allclose(compute(z,y,probability_coefficient=0),compute(z+4,y,probability_coefficient=0))
    assert not torch.allclose(compute(z,y,ranking_coefficient=0),compute(z+4,y,ranking_coefficient=0))
    assert torch.allclose(compute(z,y),compute(z.flip(1),y.flip(1)))


def test_brier_and_extreme_bf16_are_finite():
    z=torch.tensor([[-1000.,1000.]],dtype=torch.bfloat16,requires_grad=True);y=torch.tensor([[True,False]])
    value=compute(z,y);assert value.dtype==torch.float32 and torch.isfinite(value)
    value.backward();assert torch.isfinite(z.grad).all()
    ordinary=torch.tensor([[.2,-.4]])
    assert torch.allclose(compute(ordinary,y,ranking_coefficient=0,probability_coefficient=1,probability_score='brier'),
                          (ordinary.sigmoid()-y.float()).square().mean())


def test_padding_cannot_get_weight():
    with pytest.raises(ValueError,match='padding'):
        compute(torch.zeros(1,2),torch.zeros(1,2,dtype=torch.bool),real=torch.tensor([False]))
