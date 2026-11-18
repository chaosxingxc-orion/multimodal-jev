import itertools
import pytest
import torch
from tools.reports.analyze_context_credit_resampling import estimate, moments
from tools.reports.analyze_context_counterfactual_credit import exact_direction, conditional_direction


@pytest.mark.parametrize('gold', [0, 1, 2])
def test_exhaustive_two_sample_groups_have_correct_expectation(gold):
    # All 8^2 groups: weighted expectation, not a Monte Carlo tolerance.
    p = torch.tensor([.17, .56, .91], dtype=torch.float64)
    a = torch.tensor(list(itertools.product([0, 1], repeat=3)))
    indices = torch.tensor(list(itertools.product(range(8), repeat=2)))
    groups = a[indices]
    mass = torch.where(a.bool(), p, 1-p).prod(-1)
    group_mass = mass[indices].prod(-1)
    _, exact = exact_direction(p, gold)
    for value in estimate(p, gold, groups).values():
        torch.testing.assert_close((value*group_mass[:, None]).sum(0), exact, rtol=1e-12, atol=1e-14)
    expected = torch.stack([conditional_direction(p, gold, x).mean(0) for x in groups])
    torch.testing.assert_close(estimate(p, gold, groups)['local_expectation'], expected)


def test_mse_decomposition_and_zero_rate():
    values = torch.tensor([[0., 0.], [1., 2.], [2., 1.]], dtype=torch.float64)
    result = moments(values, torch.tensor([.3, .6], dtype=torch.float64))
    assert result['mse'] == pytest.approx(result['variance_trace_population']+result['empirical_mean_error_squared'])
    assert result['zero_fraction'] == pytest.approx(1/3)


def test_constant_wrong_groups():
    p = torch.tensor([.01, .9], dtype=torch.float64)
    actions = torch.tensor([[[0, 1]]*16]*4)
    result = estimate(p, 0, actions)
    assert (result['rloo'] == 0).all()
    assert (result['local_expectation'][:, 0] > 0).all()


@pytest.mark.parametrize('actions', [torch.zeros(2, 1, 2), torch.full((2, 16, 2), .5)])
def test_invalid_groups(actions):
    with pytest.raises(ValueError): estimate(torch.tensor([.2, .6]), 0, actions)
