import copy

import pytest
import torch

from tools.reports.replay_context_reward_groups import identity, replay


def fixture_query(seed=971):
    p = torch.linspace(.01, .99, 60)
    a = (torch.rand((1, 16, 60), generator=torch.Generator().manual_seed(seed)) < p).long()
    return dict(request_id='test', behavior_record=dict(prob=p.tolist()),
                rollout=dict(seed=seed, group_size=16, probability=identity(p), actions=identity(a))), a[0]


def test_reward_from_explicit_sets():
    query, actions = fixture_query()
    result = replay(query, 20)
    expected = []
    for action in actions:
        selected = set(torch.where(action)[0].tolist())
        expected.append(2 * len(selected & {20}) / (len(selected) + 1))
    assert result['rewards'] == pytest.approx(expected)
    assert result['sampled_positive_rewards'] == sum(x > 0 for x in expected)


@pytest.mark.parametrize('field', ['seed', 'actions', 'probability'])
def test_reject_corrupt_provenance(field):
    query, _ = fixture_query()
    query = copy.deepcopy(query)
    if field == 'seed':
        query['rollout'][field] += 1
    else:
        query['rollout'][field]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='identity mismatch'):
        replay(query, 20)


def test_constant_wrong_group_has_zero_sampled_direction():
    query, _ = fixture_query()
    p = torch.full((60,), 1e-12)
    p[1] = 1 - 1e-6
    a = (torch.rand((1, 16, 60), generator=torch.Generator().manual_seed(query['rollout']['seed'])) < p).long()
    query['behavior_record']['prob'] = p.tolist()
    query['rollout'].update(probability=identity(p), actions=identity(a))
    result = replay(query, 20)
    assert result['reward_kind'] == 'all_zero'
    assert all(v['l2'] == 0 for v in result['score_directions'].values())
