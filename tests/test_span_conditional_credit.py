import itertools
import pytest
import torch
from multimodal_jev.span_conditional_credit import conditional_credit,reference_conditional_loss
from multimodal_jev.span_rule_objectives import expected_rule_reward,rule_sample_rewards


def fixture():
    logits=torch.tensor([[[.3,-.2],[.5,.1],[-.4,.8]],[[.8,.1],[-.3,.6],[.2,-.1]]],requires_grad=True)
    actor=tuple(logits[:,j].log_softmax(-1) for j in range(3))
    old=tuple(p.detach().clone() for p in actor)
    target=torch.tensor([[1,-100,0],[0,1,1]]);mask=target!=-100
    return logits,actor,old,target,mask


@pytest.mark.parametrize('weight',[0.,.2])
def test_onpolicy_gradient_matches_exact_expected_reward(weight):
    logits,actor,old,target,mask=fixture()
    loss=reference_conditional_loss(old,actor,target,mask,weight)
    gradient=torch.autograd.grad(loss,logits,retain_graph=True)[0]
    expected=torch.autograd.grad(-expected_rule_reward(actor,target,mask,weight).mean(),logits)[0]
    torch.testing.assert_close(gradient,expected,atol=1e-7,rtol=1e-6)
    assert abs(float(loss.detach()))<1e-7
    if weight==0:assert gradient[:,:2].count_nonzero()==0


def test_exact_conditional_advantage_matches_full_action_enumeration():
    _,_,old,target,mask=fixture()
    advantages,_=conditional_credit(old,target,mask,.2)
    actions=torch.tensor(list(itertools.product((0,1),repeat=3))).expand(2,-1,-1)
    reward=rule_sample_rewards(actions,target,mask,.2)['total']
    joint=torch.stack([p.exp().gather(1,actions[:,:,j]) for j,p in enumerate(old)]).prod(0)
    expected=(reward*joint).sum(-1)
    for j in range(3):
        for value in (0,1):
            active=actions[:,:,j]==value
            conditional=(reward*joint*active).sum(-1)/(joint*active).sum(-1)
            torch.testing.assert_close(advantages[:,j,value],conditional-expected,atol=1e-7,rtol=1e-6)


def test_final_weight_invariant_to_extra_unscored_spans():
    _,actor,old,target,mask=fixture()
    changed=list(actor);changed[-1]=(actor[-1]+torch.tensor([.2,-.2])).log_softmax(-1)
    full=reference_conditional_loss(old,changed,target,mask,0.)
    single=reference_conditional_loss(old[-1:],changed[-1:],target[:,-1:],mask[:,-1:],0.)
    torch.testing.assert_close(full,single)


@pytest.mark.parametrize('fault',['grad_old','unnormalized','missing_final','bad_actor','bad_clip'])
def test_invalid_behavior_or_supervision_rejected(fault):
    _,actor,old,target,mask=fixture();old=list(old);actor=list(actor)
    if fault=='grad_old':old[0].requires_grad_()
    elif fault=='unnormalized':old[0]=old[0]+.1
    elif fault=='missing_final':mask[0,-1]=False
    elif fault=='bad_actor':actor[0]=actor[0]+.1
    with pytest.raises(ValueError):reference_conditional_loss(old,actor,target,mask,.2,1. if fault=='bad_clip' else .2)
