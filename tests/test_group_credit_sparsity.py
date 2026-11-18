import itertools
import math
from pathlib import Path
import runpy

import pytest

m=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/jev/audit_group_credit_sparsity.py'))


@pytest.mark.parametrize('p,g,w',[(1/60,16,.025),(.5,16,.1),(.9,32,0.)])
def test_mass_binomial_events_and_rloo_unbiased_gradient(p,g,w):
    r=m['enumerate_case'](p,.5,w,g)
    assert r['probability_mass']==pytest.approx(1,abs=1e-12)
    assert r['enumerated_count_vectors']==math.comb(g+3,3)
    assert r['final_all_wrong_probability']==pytest.approx((1-p)**g)
    assert r['final_constant_probability']==pytest.approx((1-p)**g+p**g)
    assert r['expected_group_logit_gradients']['rloo']==pytest.approx([p*(1-p),w/4],abs=1e-12)


def test_histograms_match_independent_ordered_group_enumeration():
    pf,pq,w,g=.2,.3,.1,3
    actions=list(itertools.product((0,1),repeat=2))
    total=[0.,0.]
    for group in itertools.product(actions,repeat=g):
        probability=math.prod((pf if f else 1-pf)*(pq if q else 1-pq) for f,q in group)
        rewards=[f+w*q for f,q in group];mean=sum(rewards)/g
        std=math.sqrt(sum((r-mean)**2 for r in rewards)/(g-1))
        for j,p in enumerate((pf,pq)):
            total[j]+=probability*sum((r-mean)/(std+1e-6)*(a[j]-p) for r,a in zip(rewards,group))/g
    r=m['enumerate_case'](pf,pq,w,g)
    assert r['expected_group_logit_gradients']['grpo']==pytest.approx(total,abs=1e-12)


def test_final_constant_group_has_no_final_score_gradient():
    for counts in [(8,8,0,0),(0,0,8,8)]:
        t=m['group_terms'](counts,.2,.5,.025)
        assert t['final_constant'] and not t['total_reward_constant']
        assert t['gradients']['grpo'][0]==pytest.approx(0,abs=1e-12)
        assert t['gradients']['grpo'][1]>0


def test_zero_process_budget_cannot_create_expected_process_gradient():
    r=m['enumerate_case'](1/60,.5,0.,16)
    assert r['expected_group_logit_gradients']['grpo'][1]==pytest.approx(0,abs=1e-12)
    assert r['total_reward_constant_probability']==pytest.approx(r['final_constant_probability'])
