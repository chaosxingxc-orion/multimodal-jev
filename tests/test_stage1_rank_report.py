from tools.reports.stage1_top3_misses import inspect_rank


def test_third_place_ties_distinguished_from_definite_misses():
    labels = list('abcdef')
    scores = [.9, .7, .62, .62, .62, .1]
    inside, outside, definite = [inspect_rank('u', k, labels, scores) for k in 'cef']
    assert [r['rank'] for r in (inside, outside, definite)] == [3, 5, 6]
    assert (inside['rank_best'], outside['rank_worst'], definite['rank_best']) == (3, 5, 6)
