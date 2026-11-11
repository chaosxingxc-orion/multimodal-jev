import itertools
import torch
from multimodal_jev.span_credit_diagnostic import exact_additive_logit_gradient,sampled_logit_gradient
from multimodal_jev.span_rule_objectives import expected_rule_reward,rule_sample_rewards


def test_exact_logit_credit_matches_autograd_and_final_only_ignores_fact_logits():
    logits=torch.tensor([[[.2,.7],[1.,-.1],[.3,-.4]]],requires_grad=True)
    logp=tuple(logits[:,j].log_softmax(-1) for j in range(3))
    target=torch.tensor([[1,0,1]]);mask=torch.ones_like(target,dtype=torch.bool)
    for weight in (0.,.2):
        expected=expected_rule_reward(logp,target,mask,weight).sum()
        actual=torch.autograd.grad(expected,logits,retain_graph=True)[0]
        torch.testing.assert_close(actual,exact_additive_logit_gradient(logp,target,mask,weight))
    assert exact_additive_logit_gradient(logp,target,mask,0.)[:,:2].count_nonzero()==0


def test_probability_weighted_enumeration_matches_exact_score_gradient():
    p=(torch.tensor([[.2,.8]]).log(),torch.tensor([[.7,.3]]).log(),torch.tensor([[.4,.6]]).log())
    actions=torch.tensor(list(itertools.product((0,1),repeat=3))).unsqueeze(0)
    target=torch.tensor([[1,0,0]]);mask=torch.ones_like(target,dtype=torch.bool)
    reward=rule_sample_rewards(actions,target,mask,.2)['total']
    expectation=expected_rule_reward(p,target,mask,.2)
    probabilities=torch.stack([q.exp().gather(1,actions[:,:,j]) for j,q in enumerate(p)]).prod(0)
    # sampled_logit_gradient averages samples; weighting by support size turns
    # exact enumeration into a probability-weighted sum.
    estimate=sampled_logit_gradient(p,actions,(reward-expectation[:,None])*probabilities*8,torch.zeros(1))
    torch.testing.assert_close(estimate,exact_additive_logit_gradient(p,target,mask,.2),atol=1e-7,rtol=1e-6)


def test_masked_fact_has_zero_exact_credit_even_with_auxiliary_reward():
    p=tuple(torch.tensor([[.3,.7]]).log() for _ in range(3))
    t=torch.tensor([[-100,1,0]]);mask=t!=-100
    result=exact_additive_logit_gradient(p,t,mask,.2)
    assert result[:,0].count_nonzero()==0 and result[:,1].count_nonzero()==2
