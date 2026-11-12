import pytest
import torch

from multimodal_jev.span_update_diagnostic import diagnose_update
from multimodal_jev.span_conditional_credit import reference_conditional_loss
from multimodal_jev.span_rule_objectives import rule_sft_loss


def test_small_ascent_step_has_positive_reward_and_quadratic_remainder():
    logits = torch.tensor([[.2, -.1], [-.5, .7]], dtype=torch.float64, requires_grad=True)
    old = (logits.log_softmax(-1),)
    target = torch.tensor([[1], [0]])
    mask = torch.ones_like(target, dtype=torch.bool)
    loss = reference_conditional_loss(tuple(p.detach() for p in old), old, target, mask)
    gradient = torch.autograd.grad(loss, logits)[0]
    for scale in (1e-3, 5e-4):
        after = ((logits.detach()-scale*gradient).log_softmax(-1),)
        report = diagnose_update(tuple(p.detach() for p in old), after, target, mask, 0.)
        assert report['reward_delta'] > 0
        assert abs(report['nonlinear_remainder']) < scale**2
        assert report['spans'][0]['gold_true']['requests'] == 1


def test_balanced_accuracy_can_stay_flat_while_reward_improves():
    target = torch.tensor([[0], [1]])
    before = (torch.tensor([[.6,.4],[.6,.4]]).log(),)
    after = (torch.tensor([[.8,.2],[.55,.45]]).log(),)
    r = diagnose_update(before, after, target, torch.ones_like(target,dtype=torch.bool), 0.)
    assert r['reward_delta'] > 0
    assert r['spans'][0]['all']['accuracy_after'] == r['spans'][0]['all']['accuracy_before'] == .5
    assert r['spans'][0]['all']['predicted_true_after'] == 0.


def test_unsupervised_facts_do_not_contribute_and_invalid_policy_rejected():
    target = torch.tensor([[-100, -100, 1]])
    mask = target != -100
    old = tuple(torch.tensor([[.6,.4]]).log() for _ in range(3))
    new = (torch.tensor([[.1,.9]]).log(), old[1], old[-1])
    r = diagnose_update(old, new, target, mask, .2)
    assert r['reward_delta'] == 0 and r['linearized_reward_delta'] == 0
    assert r['spans'][0]['all']['requests'] == 0
    with pytest.raises(ValueError):
        diagnose_update(old, (new[0]+1,new[1],new[2]), target, mask, .2)


def test_onpolicy_conditional_gradient_is_gold_probability_scaled_sft():
    logits = torch.tensor([[[.3,-.2],[.5,.1],[-.4,.8]]], dtype=torch.float64, requires_grad=True)
    actor = tuple(logits[:,j].log_softmax(-1) for j in range(3))
    gold = torch.tensor([[1,0,1]])
    mask = torch.ones_like(gold,dtype=torch.bool)
    conditional = reference_conditional_loss(tuple(p.detach() for p in actor),actor,gold,mask,.2)
    g_conditional = torch.autograd.grad(conditional,logits,retain_graph=True)[0]
    g_sft = torch.autograd.grad(rule_sft_loss(actor,gold,mask,.2),logits)[0]
    p_gold = torch.stack(actor,1).exp().gather(2,gold[:,:,None]).detach()
    torch.testing.assert_close(g_conditional,g_sft*p_gold,atol=1e-10,rtol=1e-7)
