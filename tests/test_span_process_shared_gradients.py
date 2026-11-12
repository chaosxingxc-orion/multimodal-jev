"""Shared-coordinate controls for interpreting the real-audio CPU diagnostic."""
import pytest
import torch

from multimodal_jev.span_finite_credit import verl_loss
from diagnose_intent_shared_gradients_cpu import geometry


def test_on_policy_additive_expected_reward_matches_real_verl_shared_gradient():
    torch.manual_seed(317)
    shared=torch.randn(4,3,requires_grad=True)
    features=torch.randn(5,4)
    projections=[torch.randn(3,k) for k in (2,3,7)]
    policy=tuple((features@shared@head).log_softmax(-1) for head in projections)
    behavior=tuple(p.detach().clone() for p in policy)
    targets=torch.tensor([[0,1,6],[1,2,0],[0,0,3],[1,1,4],[0,2,2]])
    supervised=torch.ones_like(targets,dtype=torch.bool)
    weights=torch.tensor([[.1,.1,1.]]*5)
    request_weights=torch.tensor([.3,1.,2.,.5,1.7])
    expected=[]
    for j,p in enumerate(policy):
        expected.append(-(p.exp().gather(1,targets[:,j:j+1]).flatten()*weights[:,j]*request_weights).sum()/request_weights.sum())
    gf=torch.autograd.grad(expected[-1],shared,retain_graph=True)[0]
    gp=torch.autograd.grad(sum(expected[:-1]),shared,retain_graph=True)[0]
    loss=verl_loss(behavior,policy,targets,supervised,weights,request_weights)[0]
    gv=torch.autograd.grad(loss,shared)[0]
    torch.testing.assert_close(gv,gf+gp,atol=1e-7,rtol=1e-5)
    assert abs(loss.item())<1e-6  # Near-zero centered value still has gradient.
    assert gv.norm()>0
    report=geometry(dict(shared=gf),dict(shared=gp))
    assert report['combined_gradient_dot_final']==pytest.approx(((gf.double()+gp.double())*gf.double()).sum().item(),abs=1e-12)


def test_same_logit_magnitudes_do_not_determine_shared_gradient_conflict():
    f=dict(shared=torch.tensor([1.,2.]))
    aligned=geometry(f,dict(shared=torch.tensor([2.,4.])))
    opposed=geometry(f,dict(shared=torch.tensor([-2.,-4.])))
    assert aligned['process_to_final_norm']==opposed['process_to_final_norm']==2.
    assert aligned['cosine']==1. and opposed['cosine']==-1.
    assert aligned['combined_gradient_dot_final']>0>opposed['combined_gradient_dot_final']


def test_geometry_rejects_missing_or_nonfinite_coordinates():
    with pytest.raises(ValueError):geometry(dict(a=torch.ones(2)),dict(b=torch.ones(2)))
    with pytest.raises(ValueError):geometry(dict(a=torch.ones(2)),dict(a=torch.ones(3)))
    with pytest.raises(ValueError):geometry(dict(a=torch.ones(2)),dict(a=torch.tensor([float('nan'),0.])))
    zero=geometry(dict(a=torch.zeros(2)),dict(a=torch.zeros(2)))
    assert zero['cosine'] is None and zero['process_to_final_norm'] is None
