import math
import pytest
from multimodal_jev.span_credit_concentration import sensitivity, concentration


def test_expected_correctness_gradient_matches_finite_difference():
    logits = [-8., -3., 1., .5]; y = 0; weight = .4
    def probs(z):
        mass = sum(math.exp(x) for x in z); return [math.exp(x)/mass for x in z]
    expected = sensitivity([math.log(p) for p in probs(logits)], y, weight)
    for i in range(len(logits)):
        plus = logits.copy(); minus = logits.copy(); plus[i] += 1e-5; minus[i] -= 1e-5
        finite = weight*(probs(plus)[y]-probs(minus)[y])/2e-5
        assert expected['gradient'][i] == pytest.approx(finite, abs=1e-12)
    assert sum(expected['gradient']) == pytest.approx(0., abs=1e-16)


def test_low_probability_suppresses_gradient_despite_wrong_prediction():
    low = sensitivity([math.log(.0001), math.log(.9999)], 0)
    mid = sensitivity([math.log(.1), math.log(.9)], 0)
    assert mid['norm']/low['norm'] > 800
    assert sensitivity([math.log(.5), math.log(.5)], 0, 0)['norm'] == 0


def test_effective_count_and_zero_credit_are_distinct():
    rows = [dict(id=str(i), norm=1., gold_probability=.1) for i in range(10)]
    result = concentration(rows)
    assert result['effective_requests'] == pytest.approx(10)
    assert result['top2_share'] == pytest.approx(.2)
    assert concentration([dict(id='x', norm=0., gold_probability=0.)])['top2_share'] is None
    with pytest.raises(ValueError): concentration(rows+rows[:1])
