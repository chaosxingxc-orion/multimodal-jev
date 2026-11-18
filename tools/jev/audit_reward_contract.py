#!/usr/bin/env python3
"""Exact toy-policy checks for the task reward contract; not a verl integration."""
from fractions import Fraction as F
import argparse
import itertools
import json
from pathlib import Path


def fixed_reward(final_correct, correct, masks, budgets):
    """budgets maps families to (fixed number of configured Q, family cap)."""
    result=F(final_correct)
    for family,(count,budget) in budgets.items():
        if count<0 or len(correct[family])!=count or len(masks[family])!=count:
            raise ValueError('fixed configured question count required')
        if count:
            result+=budget*sum(F(c)*m for c,m in zip(correct[family],masks[family],strict=True))/count
    return result


def audit():
    # One process action q and one Final action f, independent given input.
    pq,pf=F(3,10),F(1,5)
    actions=list(itertools.product((0,1),repeat=2))
    probs={(q,f):(pq if q else 1-pq)*(pf if f else 1-pf) for q,f in actions}
    score=lambda a:(F(a[0])-pq,F(a[1])-pf)
    rewards={a:F(a[1])+F(1,5)*a[0] for a in actions}
    expected=sum(probs[a]*rewards[a] for a in actions)
    gradient=tuple(sum(probs[a]*rewards[a]*score(a)[j] for a in actions) for j in (0,1))
    conditional={}
    for j in (0,1):
        for value in (0,1):
            mass=sum(probs[a] for a in actions if a[j]==value)
            conditional[j,value]=sum(probs[a]*rewards[a] for a in actions if a[j]==value)/mass-expected
    conditional_gradient=tuple(sum(probs[a]*conditional[j,a[j]]*score(a)[j] for a in actions) for j in (0,1))
    assert gradient==conditional_gradient==(F(21,500),F(4,25))
    # Exhaustive independent two-sample groups: exact RLOO on-policy estimator.
    rloo=[F(0),F(0)]
    for a,b in itertools.product(actions,repeat=2):
        for j in (0,1):
            rloo[j]+=probs[a]*probs[b]*(
                (rewards[a]-rewards[b])*score(a)[j]+(rewards[b]-rewards[a])*score(b)[j])/2
    assert tuple(rloo)==gradient
    # Final-only gives zero exact credit to independent q, despite shared parameters in a model.
    final_only_q_gradient=sum(probs[a]*a[1]*score(a)[0] for a in actions)
    assert final_only_q_gradient==0
    budgets={'intent':(4,F(1,10)),'slot':(4,F(1,10))}
    full={'intent':[True]*4,'slot':[True]*4};mask={'intent':[True]*4,'slot':[True]*4}
    assert fixed_reward(False,full,mask,budgets)==F(1,5)
    masked={'intent':[True,False,False,False],'slot':[False]*4}
    assert fixed_reward(False,full,masked,budgets)==F(1,40)
    # Same-input ranking guarantee only, not guarantee of Final optimizer gradients.
    assert F(1)-F(1,5)==F(4,5)
    # Finite-sample G=2 standardized advantages, sample std, eps=1e-6.
    def grpo(rs):
        mean=sum(rs)/len(rs);std=(sum((r-mean)**2 for r in rs)/(len(rs)-1))**.5
        return [(r-mean)/(std+1e-6) for r in rs]
    raw=grpo([0.,1.]);scaled=grpo([0.,.025])
    return dict(schema='jev-reward-contract-toy-audit-v1',
        expected_reward=str(expected),exact_logit_gradient=list(map(str,gradient)),
        conditional_gradient=list(map(str,conditional_gradient)),rloo_G2_expected_gradient=list(map(str,rloo)),
        final_only_process_gradient=str(final_only_q_gradient),fixed_budget_all_correct_process='1/5',
        one_of_eight_valid_process_max='1/40',mean_valid_counterexample='1/5 (8x larger than fixed denominator)',
        same_input_final_correct_minimum_margin='4/5',grpo_unit_scale=raw,grpo_scale_point025=scaled,
        grpo_scale_warning='Group std nearly cancels small raw process reward when Final is constant; epsilon prevents exact scale invariance.',
        scope='Exact independent Bernoulli policy algebra and G2 expectation, no clipping away from old policy, no shared-backbone gradient or empirical efficacy claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=audit();a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
