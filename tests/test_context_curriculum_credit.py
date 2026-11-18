import itertools
import json
import math
from pathlib import Path

import pytest
import torch

from multimodal_jev.context_curriculum_credit import (
    advantages,
    brier_reward,
    composite,
    margin_reward,
    plackett_luce_log_prob,
    rank_tiers,
    schedule,
    surrogate_terms,
    trajectory,
)


ROOT = Path(__file__).resolve().parents[1]
MENU = torch.tensor([0.35, 0.20, 0.15, 0.12, 0.10, 0.08], dtype=torch.float32)
LABELS = torch.tensor([True, False, False, False, False, False])
MENU8 = torch.tensor([0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04], dtype=torch.float32)
LABELS8 = torch.tensor([True, False, False, False, False, False, False, False])


def test_schedule_endpoints_exact_convex_and_monotone():
    assert schedule(0, 212) == {'supervised': 0.5, 'contrast': 0.3, 'ranking': 0.2}
    assert schedule(212, 212) == {'supervised': 0.0, 'contrast': 0.4, 'ranking': 0.6}
    previous = None
    for update in range(213):
        weights = schedule(update, 212)
        values = (weights['supervised'], weights['contrast'], weights['ranking'])
        assert all(math.isfinite(v) and v >= 0 for v in values)
        assert abs(sum(values) - 1.0) <= 1e-12
        if previous is not None:
            assert values[0] <= previous[0]
            assert values[1] >= previous[1] and values[2] >= previous[2]
        previous = values


@pytest.mark.parametrize('kwargs', [
    dict(update=-1, total=212),
    dict(update=213, total=212),
    dict(update=0, total=0),
    dict(update=True, total=212),
    dict(update=0.5, total=212),
    dict(update=0, total=212, name='not_registered'),
    dict(update=0, total=212, start=(0.6, 0.3, 0.2)),
    dict(update=0, total=212, start=(0.5, 0.3)),
    dict(update=0, total=212, end=(0.0, 0.5, 0.6)),
    dict(update=0, total=212, start=(-0.1, 0.5, 0.6)),
])
def test_schedule_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        schedule(**kwargs)


def test_rank_tiers_support_reproducibility_and_seed_sensitivity():
    values = set()
    first = None
    for seed in (317, 318, 319):
        for query in ('q0', 'q1'):
            out = rank_tiers(MENU, LABELS, group_size=32, seed=seed, update=7,
                             query_id=query, k=6)
            assert out['reward'].shape == (32,)
            assert 0 <= out['top1'] <= out['top3'] <= 32
            values.update(out['reward'].tolist())
            if first is None:
                first = out
    assert values <= {0.0, 0.5, 1.0}
    again = rank_tiers(MENU, LABELS, group_size=32, seed=317, update=7, query_id='q0', k=6)
    assert torch.equal(again['reward'], first['reward'])
    assert torch.equal(again['draws'], first['draws'])
    changed = rank_tiers(MENU, LABELS, group_size=32, seed=318, update=7, query_id='q0', k=6)
    assert not torch.equal(changed['reward'], first['reward'])


@pytest.mark.parametrize('kwargs', [dict(seed=-1), dict(update=-1), dict(query_id='')])
def test_rank_tiers_rejects_invalid_identity(kwargs):
    with pytest.raises(ValueError):
        rank_tiers(MENU, LABELS, **kwargs)


def test_brier_uses_distinct_trajectory_candidates_and_stays_bounded():
    out = brier_reward(MENU, LABELS, group_size=16, subset=3)
    assert out['picks'].shape == (16, 3)
    for row in out['picks']:
        assert len(torch.unique(row)) == 3
        assert bool(((row >= 0) & (row < len(MENU))).all())
    assert ((out['reward'] >= 0) & (out['reward'] <= 1)).all()


def test_brier_near_perfect_probabilities_approach_one():
    probabilities = torch.tensor([1 - 1e-6, 1e-6, 1 - 1e-6, 1e-6], dtype=torch.float32)
    labels = torch.tensor([True, False, True, False])
    out = brier_reward(probabilities, labels, group_size=4, subset=4)
    assert (out['reward'] > 0.9999).all()


@pytest.mark.parametrize('kwargs', [dict(subset=0), dict(subset=len(MENU) + 1), dict(group_size=0)])
def test_brier_rejects_invalid_sizes(kwargs):
    with pytest.raises(ValueError):
        brier_reward(MENU, LABELS, **kwargs)


@pytest.mark.parametrize('function, kwargs', [
    (brier_reward, dict(subset=2)),
    (margin_reward, dict(negatives=3)),
])
def test_brier_and_margin_are_reproducible_and_seed_sensitive(function, kwargs):
    first = function(MENU, LABELS, seed=1, update=2, query_id='same', **kwargs)
    again = function(MENU, LABELS, seed=1, update=2, query_id='same', **kwargs)
    changed = function(MENU, LABELS, seed=3, update=2, query_id='same', **kwargs)
    assert torch.equal(first['reward'], again['reward'])
    assert not torch.equal(first['reward'], changed['reward'])


def test_margin_reward_is_monotone_in_the_margin():
    rewards = [
        margin_reward(MENU, LABELS, group_size=64, negatives=4, margin=margin,
                      seed=5, update=3, query_id='margin')['reward']
        for margin in (0.0, 0.2, 0.5, 0.9)
    ]
    for lower, higher in zip(rewards, rewards[1:]):
        assert torch.all(higher <= lower + 1e-7)


@pytest.mark.parametrize('kwargs', [
    dict(negatives=0),
    dict(negatives=99),
    dict(margin=-0.1),
    dict(margin=1.0),
    dict(group_size=0),
])
def test_margin_rejects_invalid_sampling(kwargs):
    with pytest.raises(ValueError):
        margin_reward(MENU, LABELS, **kwargs)


def test_composite_convex_weights_stay_in_unit_interval():
    weights = schedule(106, 212)
    terms = dict(
        margin=torch.tensor([0.0, 0.5, 1.0, 0.25], dtype=torch.float32),
        brier=torch.tensor([1.0, 0.5, 0.0, 0.75], dtype=torch.float32),
        rank=torch.tensor([0.0, 0.5, 1.0, 0.0], dtype=torch.float32),
    )
    out = composite(dict(terms, weights=weights))
    expected = 0.35 * (0.5 * terms['margin'] + 0.5 * terms['brier']) + 0.4 * terms['rank']
    torch.testing.assert_close(out, expected)
    assert ((out >= 0) & (out <= 1)).all()
    with pytest.raises(ValueError):
        composite(dict(terms, contrast_fraction=0.0))


@pytest.mark.parametrize('bad_weights', [
    {'supervised': 0.5, 'contrast': 0.3},
    {'supervised': 0.5, 'contrast': 0.3, 'ranking': 0.3},
    {'supervised': -0.1, 'contrast': 0.5, 'ranking': 0.6},
])
def test_composite_rejects_non_convex_weights(bad_weights):
    terms = dict(
        weights=bad_weights,
        margin=torch.tensor([0.5], dtype=torch.float32),
        brier=torch.tensor([0.5], dtype=torch.float32),
        rank=torch.tensor([0.5], dtype=torch.float32),
    )
    with pytest.raises(ValueError):
        composite(terms)


def test_advantages_constant_group_is_zero_and_nonbinary_is_accepted():
    constant = torch.full((2, 8), 0.42, dtype=torch.float32)
    torch.testing.assert_close(advantages(constant), torch.zeros_like(constant))
    graded = torch.tensor([[0.1, 0.5, 0.9, 0.2], [0.3, 0.3, 0.3, 0.3]], dtype=torch.float32)
    out = advantages(graded)
    assert out.shape == graded.shape and torch.isfinite(out).all()
    assert not torch.equal(out[0], torch.zeros_like(out[0]))
    assert torch.equal(out[1], torch.zeros_like(out[1]))


@pytest.mark.parametrize('bad', [
    torch.tensor([[1.1, 0.5]]),
    torch.tensor([[-0.01, 0.5]]),
    torch.tensor([[0.1, 0.2]], requires_grad=True),
    torch.tensor([[0.1, 0.2]], dtype=torch.float64),
    torch.tensor([0.1, 0.2]),
    torch.tensor([[0.5]]),
    torch.tensor([[float('nan'), 0.5]]),
])
def test_advantages_rejects_out_of_contract_rewards(bad):
    with pytest.raises(ValueError):
        advantages(bad)


def test_moving_gold_to_the_worst_candidate_never_raises_top1():
    probabilities = torch.tensor([0.7, 0.1, 0.1, 0.1], dtype=torch.float32)
    best = torch.tensor([True, False, False, False])
    worst = torch.tensor([False, False, False, True])
    high = rank_tiers(probabilities, best, group_size=256, seed=317, update=0, query_id='disc')
    low = rank_tiers(probabilities, worst, group_size=256, seed=317, update=0, query_id='disc')
    assert low['top1'] <= high['top1']
    assert low['top1'] < high['top1']
    assert low['top3'] <= high['top3']


def test_trajectory_returns_one_shared_draw_for_all_rewards():
    common = dict(k=4, group_size=8, seed=7, update=2, query_id='shared')
    out = trajectory(MENU8, LABELS8, **common)
    rank = rank_tiers(MENU8, LABELS8, **common)
    brier = brier_reward(MENU8, LABELS8, **common)
    margin = margin_reward(MENU8, LABELS8, **common)
    for other in (rank, brier, margin):
        assert torch.equal(out['draws'], other['draws'])
    assert torch.equal(out['rank'], rank['reward'])
    assert torch.equal(out['brier'], brier['reward'])
    assert torch.equal(out['margin'], margin['reward'])
    p = MENU8.double()
    for row, rank_reward, brier_reward_value in zip(out['draws'], out['rank'], out['brier']):
        top1 = bool(LABELS8[row[:1]].any())
        top3 = bool(LABELS8[row[:3]].any())
        expected_rank = 1.0 if top1 else 0.5 if top3 else 0.0
        expected_brier = 1.0 - ((p[row] - LABELS8[row].double()) ** 2).mean()
        assert rank_reward.item() == expected_rank
        assert brier_reward_value.item() == pytest.approx(float(expected_brier), abs=1e-6)


def test_trajectory_log_prob_shape_dtype_and_detach():
    out = trajectory(MENU8, LABELS8, k=4, group_size=8, seed=7, update=2, query_id='contract')
    assert out['draws'].dtype == torch.long and out['draws'].shape == (8, 4)
    assert out['log_prob'].dtype == torch.float64
    assert out['log_prob'].shape == (8,) and not out['log_prob'].requires_grad
    for name in ('rank', 'brier', 'margin'):
        assert out[name].dtype == torch.float32 and out[name].shape == (8,)
        assert ((out[name] >= 0) & (out[name] <= 1)).all()
    assert len(out['margin_reason']) == 8


def test_trajectory_rewards_and_log_prob_are_detached_even_for_grad_weights():
    weights = MENU8.clone().requires_grad_()
    out = trajectory(weights, LABELS8, k=4, group_size=4, seed=1, update=0, query_id='detach')
    for name in ('log_prob', 'rank', 'brier', 'margin'):
        assert not out[name].requires_grad


def test_log_prob_matches_hand_computed_plackett_luce_probability():
    weights = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32)
    labels = torch.tensor([True, False, False])
    out = trajectory(weights, labels, k=2, group_size=4, seed=4, update=0, query_id='hand')
    p = weights.double()
    total = p.sum()
    for row, log_prob in zip(out['draws'], out['log_prob']):
        first, second = row.tolist()
        expected = torch.log(p[first] / total) + torch.log(p[second] / (total - p[first]))
        torch.testing.assert_close(log_prob, expected)
    torch.testing.assert_close(out['log_prob'], plackett_luce_log_prob(weights, out['draws']))


def test_log_prob_increases_when_gold_weight_rises():
    low = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float32)
    high = torch.tensor([0.7, 0.15, 0.1, 0.05], dtype=torch.float32)
    gold_first = torch.tensor([[0, 1, 2]])
    assert plackett_luce_log_prob(high, gold_first) > plackett_luce_log_prob(low, gold_first)
    sequences = torch.tensor(list(itertools.permutations(range(4), 3)))
    gold_mask = (sequences == 0).any(dim=1)
    low_mass = torch.logsumexp(plackett_luce_log_prob(low, sequences)[gold_mask], 0)
    high_mass = torch.logsumexp(plackett_luce_log_prob(high, sequences)[gold_mask], 0)
    assert high_mass > low_mass


def test_margin_reward_matches_hand_computed_trajectory_pairs():
    out = trajectory(MENU8, LABELS8, k=5, group_size=16, margin=0.2,
                     seed=11, update=1, query_id='pairs')
    p = MENU8.double()
    for row, reward, reason in zip(out['draws'], out['margin'].tolist(), out['margin_reason']):
        golds = [int(i) for i in row if LABELS8[i]]
        negatives = [int(i) for i in row if not LABELS8[i]]
        if not golds:
            assert reward == 0.0 and reason == 'no_gold_in_trajectory'
        elif not negatives:
            assert reward == 0.0 and reason == 'no_non_gold_in_trajectory'
        else:
            expected = sum(float(p[g] - p[n]) >= 0.2 for g in golds for n in negatives)
            expected /= len(golds) * len(negatives)
            assert reward == pytest.approx(expected, abs=1e-6)
            assert reason == 'ok'


def test_margin_empty_gold_branch_is_zero_and_reported():
    out = trajectory(MENU8, torch.zeros_like(LABELS8), k=4, group_size=4,
                     seed=3, update=0, query_id='no-gold')
    assert (out['margin'] == 0).all()
    assert set(out['margin_reason']) == {'no_gold_in_trajectory'}
    assert (out['rank'] == 0).all()
    wrapped = margin_reward(MENU8, torch.zeros_like(LABELS8), k=4, group_size=4,
                            seed=3, update=0, query_id='no-gold')
    assert set(wrapped['margin_reason']) == {'no_gold_in_trajectory'}


def test_margin_empty_non_gold_branch_is_zero_and_reported():
    out = trajectory(MENU8, torch.ones_like(LABELS8), k=4, group_size=4,
                     seed=3, update=0, query_id='all-gold')
    assert (out['margin'] == 0).all()
    assert set(out['margin_reason']) == {'no_non_gold_in_trajectory'}
    assert (out['rank'] == 1).all() and out['top1'] == 4 and out['top3'] == 4


@pytest.mark.parametrize('kwargs', [dict(k=0), dict(k=len(MENU8) + 1),
                                   dict(k=3, group_size=1), dict(k=3, margin=1.0)])
def test_trajectory_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        trajectory(MENU8, LABELS8, **kwargs)


def test_surrogate_terms_center_composite_scores_and_stay_detached():
    out = trajectory(MENU8, LABELS8, k=4, group_size=8, seed=2, update=0, query_id='surrogate')
    scores = composite(dict(weights=schedule(0, 212), margin=out['margin'],
                            brier=out['brier'], rank=out['rank']))
    terms = surrogate_terms(out['log_prob'], scores)
    assert terms['advantages'].shape == scores.shape
    assert not terms['advantages'].requires_grad and not terms['log_prob'].requires_grad
    assert torch.equal(terms['scores'], scores)
    constant = torch.full_like(scores, 0.4)
    centered = surrogate_terms(out['log_prob'], constant)['advantages']
    torch.testing.assert_close(centered, torch.zeros_like(scores))


@pytest.mark.parametrize('log_prob, scores', [
    (torch.tensor([0.1, -0.2], dtype=torch.float64), torch.tensor([0.5, 0.5])),
    (torch.tensor([-0.1], dtype=torch.float64), torch.tensor([0.5, 0.5])),
    (torch.tensor([-0.1, -0.2], dtype=torch.float64, requires_grad=True), torch.tensor([0.5, 0.5])),
])
def test_surrogate_terms_rejects_invalid_contracts(log_prob, scores):
    with pytest.raises(ValueError):
        surrogate_terms(log_prob, scores)


def test_preregistered_config_matches_module_and_replica_topology():
    config = json.loads((ROOT / 'configs/experiments/context-annealed-curriculum-v1.json').read_text())
    replica = json.loads((ROOT / 'configs/experiments/context-replica-training-dp6-formal-v1.json').read_text())
    assert config['schema'] == 'jev-curriculum-training-design-v1'
    assert config['status']['phase'] == 'preregistered_not_dispatched'
    assert config['training_dispatch_allowed'] is False
    assert config['algorithm'] == 'curriculum_grpo_floor_clipped_v1'
    assert config['topology'] == replica['topology']
    schedule_config = config['schedule']
    assert schedule_config['total_updates'] == 212
    keys = ('supervised', 'contrast', 'ranking')
    assert schedule(0, 212) == dict(zip(keys, schedule_config['start']))
    assert schedule(212, 212) == dict(zip(keys, schedule_config['end']))
    constants = config['constants']
    assert constants['trajectory_k'] == 8
    assert constants['margin'] == 0.2
    assert constants['contrast_fraction'] == 0.5
    assert constants['std_floor'] == 0.25
    assert constants['clip_ratio'] == 0.2
    assert constants['dual_clip_c'] == 3.0
    assert config['trajectory']['k'] == 8
    assert 'log_prob' in config['trajectory']
    assert config['rewards']['ranking']['trajectory_k'] == 8
    assert config['rewards']['ranking']['tier_positions'] == 3
    assert config['rewards']['ranking']['tiers'] == {
        'top1_hit': 1.0, 'top3_hit': 0.5, 'otherwise': 0.0}
    assert config['rewards']['probability']['labels_enter_draw'] is False
    assert config['rewards']['contrast']['trajectory_k'] == 8
    assert any('margin_reason' in issue for issue in config['open_issues'])
    assert any('SFT internal' in issue for issue in config['open_issues'])
    assert config['removed_supervision'] == ['binary_nll 0.3', 'supervised_ranking 0.1']
    assert config['evaluation']['evaluation_steps'] == replica['evaluation']['evaluation_steps']
    assert config['evaluation']['evaluation_steps'] == [0, 21, 42, 64, 85, 106, 127, 148, 170, 191, 212]
    assert config['engineering']['module'] == 'src/multimodal_jev/context_curriculum_credit.py'
    assert config['engineering']['candidate_module'] == 'tools/reports/context_curriculum_candidate.py'
