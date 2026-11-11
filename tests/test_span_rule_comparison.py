from copy import deepcopy
import pytest
from test_span_rule_balance import fixture
from multimodal_jev.span_rule_comparison import compare_predictions


def predictions(rows,perfect):
    return [dict(id=r['id'],predictions=r['targets'] if perfect else [False]*len(r['targets']),
        request_identity=dict(policy_id='new' if perfect else 'old',request=r['request'],features='unchanged')) for r in rows]


def test_paired_gains_cover_all_requests_without_claiming_unseen_initialization():
    rows=fixture();before=predictions(rows,False);after=predictions(rows,True)
    training=[r for r in rows if r['arm']=='structured'][:2]
    report=compare_predictions(rows,before,after,training)
    for arm in ('structured','reordered','final_only'):
        full=next(g for g in report['groups'] if g['arm']==arm and g['category']=='all')
        subsets=[g for g in report['groups'] if g['arm']==arm and g['category']!='all']
        assert sum(g['n'] for g in subsets)==full['n']
        assert full['before_accuracy']==.5 and full['after_accuracy']==1
        assert full['wins']==full['n']//2 and full['losses']==0
        assert full['before_macro_rule_ba']==.5 and full['after_macro_rule_ba']==1
    assert 'initializer previously saw' in report['scope']


@pytest.mark.parametrize('fault',['missing','duplicate','changed_input','non_boolean'])
def test_reject_incomplete_or_noncomparable_predictions(fault):
    rows=fixture();before=predictions(rows,False);after=deepcopy(predictions(rows,True))
    if fault=='missing':after.pop()
    elif fault=='duplicate':after.append(deepcopy(after[0]))
    elif fault=='changed_input':after[0]['request_identity']['features']='different'
    else:after[0]['predictions'][0]=1
    with pytest.raises(ValueError):compare_predictions(rows,before,after,rows[:1])
