import pytest
import torch
from multimodal_jev.context_tpo_contrast import ContrastSpec, project
from multimodal_jev.context_tpo_rewards import projection
from multimodal_jev.frozen_probability import transform, validate

def test_rank_independent_probability_target_and_gradients():
    p=torch.tensor([[.7,.8],[.6,.9]],dtype=torch.float64)
    y=torch.tensor([0,1]); spec=ContrastSpec(1.,1e-6,1.,.5,.5,1.)
    scale=dict(a=3.,b=-2.,rank_weight=1.,probability_weight=1.)
    (rank,q),_=project(p,y,spec,scale)
    (rank2,q2),_=project(p,y,spec,dict(scale,b=2.))
    assert torch.equal(rank,rank2) and not torch.equal(q,q2)
    z=p.logit().float().requires_grad_()
    loss=projection(transform(z,scale),q).sum()
    loss.backward()
    assert (z.grad.gather(1,y[:,None])<0).all()
    assert (z.grad.gather(1,(1-y)[:,None])>0).all()
    expected=scale['a']*(transform(z.detach(),scale).sigmoid()-q.sigmoid())/p.shape[1]
    torch.testing.assert_close(z.grad,expected)
    with pytest.raises(ValueError):validate(dict(scale,a=0.))
