import copy
import itertools
import numpy as np
import pytest
from tools.reports.context_attribute_paired_bootstrap import aggregate,replicate_means,compare_records


def pair():
    a=[];b=[]
    for i,g in enumerate(('a','a','a','b','c')):
        row=dict(id=str(i),source_group=g,gold=g,structural_slice='slice',set_f1=0.,exact_set=False,top1_correct=False)
        a.append(row);b.append(dict(row,set_f1=1. if g=='a' else 0.,exact_set=g=='a',top1_correct=g=='a'))
    return a,b


def test_all_cluster_draws_match_independent_record_duplication():
    a,b=pair();names,sums,counts=aggregate(a,b)
    draws=np.asarray(list(itertools.product(range(3),repeat=3)),dtype=np.int64)
    actual=replicate_means(sums,counts,draws)
    expected=[]
    for draw in draws:
        values=[y['set_f1']-x['set_f1'] for g in draw for x,y in zip(a,b) if x['source_group']==names[g]]
        expected.append(sum(values)/len(values))
    np.testing.assert_allclose(actual[:,0],expected,atol=0,rtol=0)
    assert sums[:,0].sum()/counts.sum()==.6
    assert np.mean(sums[:,0]/counts)!=.6  # Equal-source averaging would change estimand.


def test_determinism_swap_and_zero_control():
    a,b=pair();r=compare_records(a,b,endpoint='fixed_step634')
    assert r==compare_records(a[::-1],b[::-1],endpoint='fixed_step634')
    assert r['metrics']['set_f1']['delta']==.6
    opposite=compare_records(b,a,endpoint='fixed_step634')
    np.testing.assert_allclose(opposite['metrics']['set_f1']['percentile_95'],[-v for v in r['metrics']['set_f1']['percentile_95'][::-1]])
    for m in compare_records(a,a,endpoint='fixed_step634')['metrics'].values():assert m==dict(delta=0.,percentile_95=[0.,0.])


@pytest.mark.parametrize('fault',['duplicate','group','truth','nan','boolean','one_group','endpoint'])
def test_invalid_pair_rejected(fault):
    a,b=pair();endpoint='fixed_step634'
    if fault=='duplicate':a[1]=copy.deepcopy(a[0]);b[1]=copy.deepcopy(b[0])
    if fault=='group':b[0]['source_group']='wrong'
    if fault=='truth':a[1]['gold']=b[1]['gold']='different'
    if fault=='nan':b[0]['set_f1']=float('nan')
    if fault=='boolean':b[0]['exact_set']=1
    if fault=='one_group':a=a[:3];b=b[:3]
    if fault=='endpoint':endpoint='best_bootstrap'
    with pytest.raises(ValueError):compare_records(a,b,endpoint=endpoint)
