import copy
import math

import pytest

from tools.jev.context_metric_equivalence import compare


def sample():
    return dict(policy_id='original', metrics=dict(queries=9226,
        candidate_binary_nll=0.01874342859958838,
        candidate_brier=0.0046499592341003315, query_macro_set_f1=.738))


def test_actual_cross_platform_rounding():
    a=sample();b=copy.deepcopy(a)
    b['metrics'].update(candidate_binary_nll=.018743428599588375,
                        candidate_brier=.004649959234100332)
    b['input_receipts_sha256']='separately-authenticated'
    assert len(compare(a,b)['rounding_differences'])==2


@pytest.mark.parametrize('field,value',[
    ('queries',9225),('queries',9226.),('candidate_binary_nll',.018744),
    ('candidate_brier',float('nan')),('candidate_binary_nll',float('inf')),
    ('query_macro_set_f1',math.nextafter(.738,1.)),
])
def test_reject_changes(field,value):
    a=sample();b=copy.deepcopy(a);b['metrics'][field]=value
    with pytest.raises(ValueError): compare(a,b)


def test_identity_and_schema_stay_exact():
    a=sample();b=copy.deepcopy(a);b['policy_id']='different'
    with pytest.raises(ValueError):compare(a,b)
    b=copy.deepcopy(a);del b['metrics']['queries']
    with pytest.raises(ValueError):compare(a,b)


def test_exact_ulp_boundary():
    a=sample();b=copy.deepcopy(a)
    for _ in range(8):b['metrics']['candidate_brier']=math.nextafter(b['metrics']['candidate_brier'],1.)
    assert compare(a,b)['passed']
    b['metrics']['candidate_brier']=math.nextafter(b['metrics']['candidate_brier'],1.)
    with pytest.raises(ValueError):compare(a,b)
