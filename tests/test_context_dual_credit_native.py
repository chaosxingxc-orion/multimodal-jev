"""Requires installed pinned verl; skipped locally, mandatory in fixed image."""
import pytest
import torch

from multimodal_jev.context_dual_credit import hybrid_loss, prepare


def test_native_full_vs_dp_micro_and_zero_padding():
    pytest.importorskip('verl')
    pytest.importorskip('omegaconf')
    z=torch.tensor([[.7,-.4,1.2],[-.2,.4,.8],[.1,-.3,.6]],requires_grad=True)
    labels=torch.tensor([[True,False,False],[False,True,False],[False,False,True]])
    ids=[dict(policy_id='p0',input_id=f'i{i}',query_id=f'q{i}',update=0) for i in range(3)]
    records=[prepare(z[i].detach(),z[i].detach().sigmoid(),['a','b','c'],labels[i],**ids[i]) for i in range(3)]
    ref=torch.full_like(z,.5).detach();real=torch.tensor([True,True,False])
    total,_=hybrid_loss(z,records,ids,ref,real,global_real_requests=2)
    gradient,=torch.autograd.grad(total,z)
    gradients=[]
    for i in range(3):
        local=z.detach().clone().requires_grad_()
        value,_=hybrid_loss(local[i:i+1],records[i:i+1],ids[i:i+1],ref[i:i+1],real[i:i+1],
                            global_real_requests=2,world_size=3)
        gradients.append(torch.autograd.grad(value,local)[0])
    torch.testing.assert_close(sum(gradients)/3,gradient)
    assert torch.equal(gradient[2],torch.zeros(3))


def test_native_surrogate_matches_independent_clipped_equation():
    pytest.importorskip('verl')
    pytest.importorskip('omegaconf')
    from multimodal_jev.context_dual_credit import native_surrogate
    old=torch.tensor([[-.8,-.3],[-.6,-.9]])
    new=(old+torch.tensor([[.4,-.5],[.2,-.2]])).requires_grad_()
    adv=torch.tensor([[1.,-1.],[.5,-.5]]);weights=torch.tensor([.75,.25])
    ratio=(new-old).clamp(-20,20).exp()
    clipped=torch.maximum(-adv*ratio,-adv*ratio.clamp(.8,1.2))
    dual=torch.where(adv<0,torch.minimum(-adv*3,clipped),clipped)
    expected=(dual.mean(-1)*weights).sum()
    actual=native_surrogate(dict(new_logprob=new,old_logprob=old,advantages=adv,weight=weights))
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(torch.autograd.grad(actual,new,retain_graph=True)[0],
                               torch.autograd.grad(expected,new)[0])


def test_native_large_pair_gap_keeps_fp32_loss_and_finite_gradients():
    pytest.importorskip('verl');pytest.importorskip('omegaconf')
    z=torch.tensor([[4.,-13.0625]],requires_grad=True)
    identity=dict(policy_id='p0',input_id='large-gap',query_id='large-gap',update=0)
    record=prepare(z[0].detach(),z[0].detach().sigmoid(),['a','b'],torch.tensor([True,False]),**identity)
    assert record['branches']['ranking']['probability'].dtype==torch.float64
    value,_=hybrid_loss(z,[record],[identity],torch.full_like(z,.5).detach(),torch.tensor([True]),global_real_requests=1)
    assert value.dtype==torch.float32 and torch.isfinite(value)
    grad,=torch.autograd.grad(value,z)
    assert torch.isfinite(grad).all() and (grad!=0).any()
