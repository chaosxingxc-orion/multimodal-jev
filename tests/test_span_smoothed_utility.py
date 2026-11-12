import math
import pytest
from multimodal_jev.span_smoothed_utility import utility_credit


def test_bounds_and_finite_credit_at_zero():
    for k in (2,31,60):
        a,b=utility_credit(0.,k),utility_credit(1.,k)
        assert a['utility']==pytest.approx(-1) and b['utility']==0
        assert a['scale']==a['maximum_scale']>b['scale']>0
        assert a['tau']==1/k**2


def test_derivative_matches_finite_difference_without_normalizing_away_scale():
    for p in (.0001,.01,.5):
        a=utility_credit(p,60);eps=1e-8
        derivative=(utility_credit(p+eps,60)['utility']-utility_credit(p-eps,60)['utility'])/(2*eps)
        assert a['scale']==pytest.approx(derivative,rel=1e-6)
    # Both examples below1/K still receive different, non-unit scales.
    assert utility_credit(.0001,60)['scale']>utility_credit(.001,60)['scale']>1


def test_low_probability_gradient_still_vanishes_at_extreme_limit():
    k=60
    assert 1e-12*utility_credit(1e-12,k)['scale']<1e-6
    # This soft utility does not magically remove all starvation.
    assert utility_credit(.1,k)['utility']>utility_credit(.01,k)['utility']


@pytest.mark.parametrize('p,k',[(float('nan'),60),(-1,60),(2,60),(.1,1),(.1,True)])
def test_invalid_inputs_rejected(p,k):
    with pytest.raises(ValueError):utility_credit(p,k)
