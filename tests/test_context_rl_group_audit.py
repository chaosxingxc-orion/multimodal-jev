import copy

import pytest
import torch

from tools.jev.context_rl_group_audit import audit, expected_advantages
from test_context_rollout_audit import evidence


def sample(algorithm):
    entry, group = evidence()
    act = group['actions'].bool();truth = group['labels'][:, None, :]
    rewards = 2*(act & truth).sum(-1).float()/(act.sum(-1)+truth.sum(-1))
    group['rewards'] = rewards
    mean = rewards.mean(-1, keepdim=True)
    group['advantages'] = ((rewards-mean)/(rewards.std(-1, keepdim=True)+1e-6)
                           if algorithm == 'grpo_v1' else (rewards-mean)*16/15)
    group['old_joint'] = sum(p.gather(1, group['actions'][:, :, j]) for j,p in enumerate(group['behavior']))
    target = dict(labels=group['labels'][0].tolist(), real_request=True, weight=1.)
    return dict(groups=[group], receipts=[entry], targets={group['request_id']:target},
                algorithm=algorithm, update=0, policy_id='p0', reference_policy_id='ref')


@pytest.mark.parametrize('algorithm', ['grpo_v1', 'rloo_v1'])
def test_both_estimators_and_real_denominator(algorithm):
    args = sample(algorithm)
    result = audit(**args)
    assert result['real_queries'] == result['checked_slots'] == 1
    assert result['padding_slots'] == 0 and result['mean_reward'] > 0


@pytest.mark.parametrize('fault', ['reward', 'estimator', 'likelihood', 'weight', 'truth',
                                  'observed', 'duplicate', 'missing_receipt', 'action', 'reference'])
def test_wrong_saved_training_evidence_rejected(fault):
    args = sample('rloo_v1');g = args['groups'][0]
    if fault == 'reward':g['rewards'][0, 0] += .01
    elif fault == 'estimator':g['advantages'] *= 2
    elif fault == 'likelihood':g['old_joint'][0, 0] += .01
    elif fault == 'weight':g['request_weight'] = 2.
    elif fault == 'truth':g['labels'][0, 1] = True
    elif fault == 'observed':g['observed'][0, 1] = False
    elif fault == 'duplicate':args['groups'].append(copy.deepcopy(g))
    elif fault == 'missing_receipt':args['receipts'] = []
    elif fault == 'action':g['actions'][0, 0, 1] = 2
    elif fault == 'reference':args['reference_policy_id'] = 'wrong'
    with pytest.raises(ValueError):audit(**args)


def test_all_padding_is_not_learning_batch():
    args = sample('grpo_v1');g = args['groups'][0]
    g['real_request'] = False;g['request_weight'] = 0.
    args['targets'][g['request_id']].update(real_request=False, weight=0.)
    with pytest.raises(ValueError, match='nonempty real'):audit(**args)


@pytest.mark.parametrize('algorithm', ['grpo_v1', 'rloo_v1'])
def test_constant_reward_keeps_zero_signal_group(algorithm):
    rewards = torch.full((1, 16), .5)
    assert torch.equal(expected_advantages(rewards, algorithm), torch.zeros_like(rewards))
