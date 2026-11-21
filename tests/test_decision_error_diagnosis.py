import pytest
from tools.reports.decision_error_diagnosis import diagnose


def test_ranking_correct_does_not_hide_extra_candidates_or_misses():
    labels=[dict(id=str(i),intent='a') for i in range(4)]
    probs=[[.9,.1,.2],[.9,.8,.1],[.4,.3,.2],[.1,.9,.8]]
    result=diagnose([dict(id=str(i),prob=p) for i,p in enumerate(probs)],labels,['a','b','c'])
    assert result['outcomes']==dict(exact=1,gold_plus_extras=1,empty=1,wrong_nonempty=1)
    assert result['top1_correct_but_set_wrong']==2
    assert result['total_false_positives']==3
    assert result['metrics']['top1_event_accuracy']==.75
    assert result['metrics']['exact_set']==.25


def test_duplicate_population_and_nonfinite_rejected():
    labels=[dict(id='a',intent='x'),dict(id='b',intent='x')]
    with pytest.raises(ValueError,match='population'):
        diagnose([dict(id='a',prob=[.2,.3])]*2,labels,['x','y'])
    with pytest.raises(ValueError,match='finite'):
        diagnose([dict(id='a',prob=[.2,float('nan')])],labels[:1],['x','y'])
