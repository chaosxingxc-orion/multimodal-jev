import copy
import math

import pytest
from tools.reports.context_probability_endpoint_uncertainty import compare, per_query
from test_context_simple_endpoint_uncertainty import fixture


def test_constant_delta_agrees_with_analytic_proper_scores():
    a,b,labels=fixture();r=compare(a,b,labels,['a','b'],['old','new'])
    expected=math.log(.2)-math.log(.8)
    for name,delta in [('candidate_binary_nll',expected),('candidate_brier',-.6)]:
        assert r['metrics'][name]['delta']==pytest.approx(delta)
        assert r['metrics'][name]['percentile_95']==pytest.approx([delta,delta])


def test_unequal_groups_remain_recording_weighted_and_order_independent():
    a,b,labels=fixture();labels[0]['source_group']='single';labels[1]['source_group']='large'
    for i in range(2,6):labels[i]['source_group']='large'
    b[0]['prob']=a[0]['prob'][:]
    r=compare(a,b,labels,['a','b'],['old','new'])
    assert r['metrics']['candidate_brier']['delta']==pytest.approx(-.6*5/6)
    reversed_result=compare(a[::-1],b[::-1],labels[::-1],['a','b'],['old','new'])
    assert reversed_result['bootstrap']==r['bootstrap']
    assert reversed_result['pairing_sha256']==r['pairing_sha256']
    for key,value in r['metrics'].items():
        other=reversed_result['metrics'][key]
        assert other['delta']==value['delta'] and other['percentile_95']==value['percentile_95']
        # The existing scorer accumulates point values in supplied order.
        assert other['initial']==pytest.approx(value['initial'],abs=1e-14,rel=0)
        assert other['final']==pytest.approx(value['final'],abs=1e-14,rel=0)


@pytest.mark.parametrize('side',['initial','final','both'])
def test_infinite_nll_not_clipped_or_dropped(side):
    a,b,labels=fixture()
    if side in ('initial','both'):a[0]['prob']=[0.,1.]
    if side in ('final','both'):b[0]['prob']=[0.,1.]
    r=compare(a,b,labels,['a','b'],['old','new'])
    nll=r['metrics']['candidate_binary_nll']
    assert nll['delta'] is None and nll['percentile_95'] is None
    assert r['metrics']['candidate_brier']['status']=='finite'
    assert r['recordings']==6


def test_correct_endpoint_probabilities_have_finite_zero_nll():
    assert per_query([1.,0.],0)==(0.,0.)


def test_self_control_zero_and_input_mismatch_rejected():
    a,_,labels=fixture();r=compare(a,a,labels,['a','b'],['old','old'])
    for value in r['metrics'].values():
        assert value['delta']==0. and value['percentile_95']==[0.,0.]
    b=copy.deepcopy(a);b[0]['request_identity']['positions']=[1,0]
    with pytest.raises(ValueError,match='inputs differ'):compare(a,b,labels,['a','b'],['old','old'])
