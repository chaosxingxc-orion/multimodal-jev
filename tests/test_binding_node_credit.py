from fractions import Fraction as F
import math
from pathlib import Path
import runpy

import pytest

m=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/jev/audit_binding_node_credit.py'))


def test_final_only_has_zero_process_credit_and_no_conditional_process_noise():
    r=m['run']()['cases']['final_B_correct_only']['nodes']
    assert [n['gradient'] for n in r]==[0,0,F(4,25)]
    assert r[0]['centered_score_variance']>0 and r[0]['conditional_score_variance']==0


def test_interacting_reward_matches_analytic_gradient_and_coordinate_variance_bound():
    p=(F(3,10),F(2,5),F(1,5))
    r=m['analyze'](p,lambda a:math.prod(a))
    for j,n in enumerate(r['nodes']):
        assert n['gradient']==p[j]*(1-p[j])*math.prod(p[k] for k in range(3) if k!=j)
        assert n['conditional_score_variance']<=n['centered_score_variance']


def test_consistency_accepts_wrong_all_false_and_suppresses_positive_B():
    r=m['run']()
    assert r['consistency_truth_table'][0]['consistent']==1
    assert r['consistency_truth_table'][0]['gold111_correct'] is False
    assert r['cases']['consistency_only']['nodes'][2]['gradient']<0


def test_marginal_frozen_credit_is_not_joint_gradient_away_from_behavior():
    r=m['run']()['frozen_old_credit_away_from_old']
    assert all(a!=b for a,b in zip(r['marginal_unclipped_gradient'],r['current_expected_reward_gradient']))
    # Numerical derivative of the actual interacting expected reward.
    for j,p in enumerate(r['new_probabilities']):
        z=math.log(float(p/(1-p)));eps=1e-6
        hi=list(map(float,r['new_probabilities']));lo=hi.copy()
        hi[j]=1/(1+math.exp(-z-eps));lo[j]=1/(1+math.exp(-z+eps))
        assert (math.prod(hi)-math.prod(lo))/(2*eps)==pytest.approx(float(r['current_expected_reward_gradient'][j]),abs=1e-9)
