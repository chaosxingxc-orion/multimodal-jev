from fractions import Fraction as F
import math
from pathlib import Path
import runpy

import pytest


m=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/jev/audit_joint_span_clipping.py'))
audit=m['audit_policy']


def test_old_policy_exact_gradients_and_zero_masked_credit():
    old=[F(1,2),F(1,2),F(1,5)]
    r=audit(old,old,[0,0,1])
    assert r['logit_gradient']==[0,0,F(4,25)]
    assert r['ratio_outside_old_mass']==r['surrogate_saturated_old_mass']==0


def test_irrelevant_coordinates_integrate_out_without_clipping():
    r=audit([F(1,2)]*8+[F(1,5)],[F(11,20)]*8+[F(11,50)],[0]*8+[1],clipped=False)
    assert r['logit_gradient']==[0]*8+[F(429,2500)]


def test_joint_clipping_exposes_reward_independent_coordinate_effect():
    old=[F(1,2)]*8+[F(1,5)];new=[F(11,20)]*8+[F(1,5)]
    r=audit(old,new,[0]*8+[1])
    assert r['surrogate_saturated_old_mass']>0
    assert r['logit_gradient'][-1]<F(4,25)
    assert any(g!=0 for g in r['logit_gradient'][:-1])
    assert m['marginalized_final'](old,new)['logit_gradient']==[F(4,25)]


def test_analytic_gradient_matches_finite_difference_away_from_kinks():
    old=[F(1,2)]*2+[F(1,5)];new=[F(11,20)]*2+[F(11,50)]
    result=audit(old,new,[0,0,1]);assert result['nondifferentiable_old_mass']==0
    for j,prob in enumerate(new):
        logit=math.log(float(prob/(1-prob)));eps=1e-6
        plus=list(new);minus=list(new)
        plus[j]=F(1/(1+math.exp(-(logit+eps))))
        minus[j]=F(1/(1+math.exp(-(logit-eps))))
        finite=float(audit(old,plus,[0,0,1])['objective']-audit(old,minus,[0,0,1])['objective'])/(2*eps)
        assert finite==pytest.approx(float(result['logit_gradient'][j]),abs=1e-8)


def test_report_has_no_clip_kink_or_unclipped_auxiliary_credit():
    report=m['run_audit']()
    assert len(report['cases'])==6
    for case in report['cases']:
        assert case['exact_joint_clipped']['nondifferentiable_old_mass']==0
        assert all(g==0 for g in case['exact_unclipped']['logit_gradient'][:-1])
