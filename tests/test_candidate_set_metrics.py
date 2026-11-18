import pytest
from multimodal_jev.candidate_set_metrics import CandidateSetMetrics


def score(truth, p):
    m=CandidateSetMetrics(.5); m.add('q', truth, p); return m.result()


def test_return_all_cannot_hide_false_candidates_behind_perfect_recall():
    r=score({'a':True,'b':True,'c':False}, {'a':1.,'b':1.,'c':1.})
    assert r['micro_recall_observed']==1
    assert r['micro_precision_observed']==pytest.approx(2/3)
    assert r['exact_set_accuracy_complete']==0
    assert r['binary_nll_infinite'] and r['mean_binary_nll_per_observed_query'] is None


def test_single_hit_is_not_full_set_recovery():
    r=score({'a':True,'b':True,'c':False}, {'a':.9,'b':.1,'c':.2})
    assert r['any_valid_coverage_nonempty_complete']==1
    assert r['micro_recall_observed']==.5
    assert r['exact_set_accuracy_complete']==0


def test_no_route_separate_from_nonempty_coverage():
    m=CandidateSetMetrics(.5)
    m.add('empty', {'a':False}, {'a':0.})
    m.add('nonempty', {'x':True}, {'x':0.})
    r=m.result()
    assert r['empty_set_precision_complete']==.5
    assert r['empty_set_recall_complete']==1
    assert r['any_valid_coverage_nonempty_complete']==0
    assert r['micro_precision_observed'] is None
    assert r['exact_set_accuracy_complete']==.5


def test_missing_label_not_negative_or_complete_set():
    r=score({'a':True,'b':None}, {'a':.8,'b':.9})
    assert r['counts']['fp']==0 and r['counts']['returned_unlabeled']==1
    assert r['exact_set_accuracy_complete'] is None
    assert r['mean_binary_brier_per_observed_query']==pytest.approx(.04)
    assert r['micro_precision_observed']==1  # Observed subset only, not full precision.


def test_all_missing_still_counts_population_and_return_budget():
    r=score({'a':None}, {'a':.5})
    assert r['counts']['queries']==1 and r['mean_returned_candidates']==1
    assert r['micro_recall_observed'] is None
    assert r['mean_binary_brier_per_observed_query'] is None


def test_order_and_id_rebinding_leave_scores_unchanged():
    assert score({'a':True,'b':False},{'a':.7,'b':.3}) == score(
        {'other':False,'new':True},{'new':.7,'other':.3})


def test_query_weighting_does_not_weight_large_menus_more():
    m=CandidateSetMetrics(.5)
    m.add('one', {'a':True}, {'a':1.})
    m.add('ten', {str(i):True for i in range(10)}, {str(i):0. for i in range(10)})
    r=m.result()
    assert r['mean_binary_brier_per_observed_query']==.5
    assert r['macro_query_observed']['recall']==.5
    assert r['micro_recall_observed']==pytest.approx(1/11)


@pytest.mark.parametrize('p',[{'a':float('nan')},{'a':1.1},{'a':True},{'wrong':.5}])
def test_invalid_prediction_fails_without_changing_population(p):
    m=CandidateSetMetrics(.5)
    with pytest.raises(ValueError):m.add('q', {'a':True}, p)
    assert not m.ids and m.counts['queries']==0


def test_duplicate_request_not_counted_twice():
    m=CandidateSetMetrics(.5);m.add('q',{'a':True},{'a':.8})
    with pytest.raises(ValueError):m.add('q',{'a':True},{'a':.8})
    assert m.result()['counts']['queries']==1
