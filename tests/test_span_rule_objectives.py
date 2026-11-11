import itertools
import pytest
import torch
from multimodal_jev.span_rule_objectives import (
    rule_sft_loss, rule_sample_rewards, expected_rule_reward,
)


def fixture():
    logp = [torch.tensor([[.2, .8], [.6, .4]]).log().requires_grad_(),
            torch.tensor([[.7, .3], [.1, .9]]).log().requires_grad_(),
            torch.tensor([[.4, .6], [.8, .2]]).log().requires_grad_()]
    targets = torch.tensor([[1, -100, 0], [0, 1, 1]])
    return logp, targets, targets != -100


def test_exact_expectation_matches_all_eight_actions_and_masking():
    logp, target, mask = fixture()
    actions = torch.tensor(list(itertools.product((0, 1), repeat=3))).expand(2, -1, -1)
    reward = rule_sample_rewards(actions, target, mask, .2)['total']
    probability = torch.stack([p.exp().gather(1, actions[:, :, i]) for i, p in enumerate(logp)]).prod(0)
    torch.testing.assert_close((reward * probability).sum(-1), expected_rule_reward(logp, target, mask, .2))
    torch.testing.assert_close(expected_rule_reward(logp, target, mask, .2), torch.tensor([.56, .35]))


def test_sft_final_only_has_no_auxiliary_gradient_and_masks_missing_gold():
    logp, target, mask = fixture()
    loss = rule_sft_loss(logp, target, mask)
    torch.testing.assert_close(loss, -(logp[2][0, 0]+logp[2][1, 1])/2)
    loss.backward()
    assert all(p.grad.count_nonzero() == 0 for p in logp[:2])
    logp, target, mask = fixture()
    rule_sft_loss(logp, target, mask, .2).backward()
    assert logp[1].grad[0].count_nonzero() == 0
    assert logp[0].grad[0, 1] == pytest.approx(-.1)
    assert logp[0].grad[1, 0] == pytest.approx(-.05)


@pytest.mark.parametrize('fault', ['missing_final', 'bad_target', 'bad_action', 'unnormalized', 'weight'])
def test_invalid_rule_supervision_or_policy_rejected(fault):
    logp, target, mask = fixture()
    if fault == 'missing_final': mask[0, -1] = False
    if fault == 'bad_target': target[0, 0] = 2
    if fault == 'unnormalized': logp[0] = logp[0] + .1
    with pytest.raises(ValueError):
        if fault == 'bad_action':
            rule_sample_rewards(torch.full((2, 4, 3), 2), target, mask)
        else:
            rule_sft_loss(logp, target, mask, 1.0 if fault == 'weight' else .2)


def test_single_question_rejects_fictitious_process_supervision():
    p = [torch.tensor([[.4, .6]]).log()]
    target = torch.tensor([[1]]); mask = torch.ones_like(target, dtype=torch.bool)
    torch.testing.assert_close(expected_rule_reward(p, target, mask), torch.tensor([.6]))
    with pytest.raises(ValueError, match='no auxiliary'):
        rule_sft_loss(p, target, mask, .2)
