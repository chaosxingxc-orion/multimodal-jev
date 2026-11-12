import itertools
import pytest
import torch
from multimodal_jev.span_finite_credit import credit,reference_loss,correctness_sampling_risk


def fixture():
    torch.manual_seed(137)
    logits=[torch.randn(3,k,requires_grad=True) for k in (60,2,5)]
    actor=tuple(x.log_softmax(-1) for x in logits);old=tuple(x.detach() for x in actor)
    targets=torch.tensor([[59,1,3],[13,-100,1],[-100,-100,-100]])
    mask=targets!=-100
    weights=torch.tensor([[1.,.1,.1],[1.,.2,.2],[1.,.1,.1]])
    requests=torch.tensor([1.,3.,7.])
    return logits,actor,old,targets,mask,weights,requests


def test_gradient_is_exact_weighted_expected_reward():
    logits,actor,old,y,mask,w,rw=fixture()
    actual=reference_loss(old,actor,y,mask,w,rw)
    grads=torch.autograd.grad(actual,logits,retain_graph=True)
    expected=0
    for j,p in enumerate(actor):
        prob=p.exp().gather(1,y[:,j:j+1].masked_fill(~mask[:,j:j+1],0)).squeeze(1)
        expected=expected-(prob*w[:,j]*mask[:,j]*rw).sum()/rw[:2].sum()
    oracle=torch.autograd.grad(expected,logits)
    for a,b in zip(grads,oracle,strict=True):
        torch.testing.assert_close(a,b,atol=2e-7,rtol=2e-5)
        assert a[2].count_nonzero()==0
    assert grads[1][1].count_nonzero()==0


def test_unscored_span_does_not_dilute_final_or_request_weight():
    _,actor,old,y,m,w,rw=fixture()
    w[:,1:]=0
    shifted=list(actor);shifted[0]=(actor[0]+torch.linspace(-1,1,60)).log_softmax(-1)
    full=reference_loss(old,shifted,y,m,w,rw)
    single=reference_loss(old[:1],shifted[:1],y[:,:1],m[:,:1],w[:,:1],rw)
    torch.testing.assert_close(full,single,atol=1e-7,rtol=1e-6)


def test_ragged_conditional_credit_matches_joint_enumeration():
    old=(torch.tensor([[.2,.3,.5]]).log(),torch.tensor([[.7,.3]]).log())
    y=torch.tensor([[2,0]]);mask=torch.ones_like(y,dtype=torch.bool);w=torch.tensor([[1.,.2]])
    a=credit(old,y,mask,w)['advantages']
    actions=torch.tensor(list(itertools.product(range(3),range(2))))
    joint=old[0].exp()[0,actions[:,0]]*old[1].exp()[0,actions[:,1]]
    reward=((actions==y).float()*w).sum(-1);mean=(joint*reward).sum()
    for j,k in enumerate((3,2)):
        for v in range(k):
            subset=actions[:,j]==v
            expected=(joint[subset]*reward[subset]).sum()/joint[subset].sum()-mean
            torch.testing.assert_close(a[j][0,v],expected,atol=1e-7,rtol=1e-6)


def test_class_weight_survives_centering():
    logits,actor,old,y,m,w,rw=fixture()
    g1=torch.autograd.grad(reference_loss(old,actor,y,m,w,torch.ones_like(rw)),logits,retain_graph=True)
    g2=torch.autograd.grad(reference_loss(old,actor,y,m,w,rw),logits)
    for a,b in zip(g1,g2,strict=True):
        torch.testing.assert_close(b[0],a[0]*.5,atol=1e-7,rtol=1e-6)
        torch.testing.assert_close(b[1],a[1]*1.5,atol=1e-7,rtol=1e-6)


def test_sixty_class_sampling_diagnostic():
    p=(torch.zeros(1,60).log_softmax(-1),);y=torch.tensor([[59]])
    risk=correctness_sampling_risk(p,y,torch.ones_like(y,dtype=torch.bool),torch.ones(1,1),32)
    assert abs(risk['all_wrong_probability'].item()-(59/60)**32)<1e-6


@pytest.mark.parametrize('fault',['grad_old','normalization','bad_target','negative_weight','grad_weight','all_missing','actor_width'])
def test_invalid_credit_rejected(fault):
    _,actor,old,y,m,w,rw=fixture();old=list(old);actor=list(actor)
    if fault=='grad_old':old[0]=old[0].requires_grad_()
    if fault=='normalization':old[0]=old[0]+.1
    if fault=='bad_target':y[0,0]=60
    if fault=='negative_weight':rw[0]=-1
    if fault=='grad_weight':w.requires_grad_()
    if fault=='all_missing':m[:]=False
    if fault=='actor_width':actor[0]=torch.zeros(3,59).log_softmax(-1)
    with pytest.raises(ValueError):reference_loss(old,actor,y,m,w,rw)
