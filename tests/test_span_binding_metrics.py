import math
import pytest
from multimodal_jev.span_binding_metrics import compare, summarize


def log(p): return [math.log(v) for v in p]


def test_semantic_following_and_symbol_following_are_distinguishable():
    mapping = {'presented_to_canonical': [1, 0]}
    semantic = compare(log([.9, .1]), log([.1, .9]), mapping)
    assert semantic['semantic_tv'] == 0 and semantic['symbol_tv'] == pytest.approx(.8)
    assert semantic['semantic_greedy_equal'] and not semantic['symbol_greedy_equal']
    symbol = compare(log([.9, .1]), log([.9, .1]), mapping)
    assert symbol['symbol_tv'] == 0 and symbol['semantic_tv'] == pytest.approx(.8)
    assert symbol['symbol_greedy_equal'] and not symbol['semantic_greedy_equal']
    mean = summarize([semantic, symbol])
    assert mean['semantic_greedy_equal'] == .5 and mean['semantic_tv'] == pytest.approx(.4)


def test_uniform_policy_does_not_claim_stable_semantic_greedy_on_ties():
    result = compare(log([.5, .5]), log([.5, .5]), {'presented_to_canonical': [1, 0]})
    assert result['semantic_tv'] == 0 and result['native_top_tied']
    assert not result['semantic_greedy_equal']


@pytest.mark.parametrize('p', [[0., 0.], [float('nan'), -1], [1000., -1], [-1000., -1000.]])
def test_malformed_policy_rejected(p):
    with pytest.raises(ValueError): compare(p, log([.5, .5]), {'presented_to_canonical': [1, 0]})
