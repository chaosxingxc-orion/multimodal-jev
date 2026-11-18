import math

import pytest

from tools.reports.context_rank_existence_diagnostics import query_scores, summarize


def test_rank_is_invariant_to_monotone_change_but_acceptance_is_not():
    a = query_scores([.4,.3,.2],0); b = query_scores([.8,.6,.4],0)
    assert a['pair_accuracy'] == b['pair_accuracy'] == 1
    assert a['false_reject'] == 1 and b['false_reject'] == 0
    assert a['false_accept_count'] == 0 and b['false_accept_count'] == 1


def test_ties_have_half_credit_and_threshold_is_inclusive():
    row = query_scores([.5,.5,.1],0)
    assert row['pair_accuracy'] == .75 and row['pair_tie_fraction'] == .5
    assert row['false_reject'] == 0 and row['false_accept_count'] == 1


def test_boundary_probabilities_are_not_clipped():
    assert query_scores([1.,0.,0.],0)['positive_nll'] == 0
    assert query_scores([0.,1.,.2],0)['positive_nll'] is None
    assert query_scores([0.,1.,.2],0)['negative_nll'] is None


def test_decomposition_reconstructs_native_candidate_weighting():
    labels=[dict(id='a',role='selection',intent='x'),dict(id='b',role='selection',intent='y')]
    records=[dict(id='a',policy_id='p',prob=[.9,.2,.3]),dict(id='b',policy_id='p',prob=[.1,.8,.4])]
    r=summarize(records,labels,['x','y','z'],'p')['means']
    assert r['candidate_brier_reconstructed'] == pytest.approx((.01+.04+.09+.01+.04+.16)/6)
    assert r['candidate_binary_nll_reconstructed'] == pytest.approx(-sum(map(math.log,[.9,.8,.7,.9,.8,.6]))/6)


@pytest.mark.parametrize('probs', [[float('nan'),.2],[float('inf'),.2],[-.1,.2],[1.1,.2],[True,.2]])
def test_invalid_probabilities_rejected(probs):
    with pytest.raises(ValueError):query_scores(probs,0)


@pytest.mark.parametrize('fault',['policy','duplicate','order','role','width'])
def test_conflicting_population_rejected(fault):
    labels=[dict(id='a',role='selection',intent='x'),dict(id='b',role='selection',intent='y')]
    records=[dict(id=r['id'],policy_id='p',prob=[.2,.8]) for r in labels]
    if fault=='policy':records[0]['policy_id']='q'
    if fault=='duplicate':labels[1]['id']='a'
    if fault=='order':records.reverse()
    if fault=='role':labels[0]['role']='learning'
    if fault=='width':records[0]['prob']=[.2]
    with pytest.raises(ValueError):summarize(records,labels,['x','y'],'p')
