#!/usr/bin/env python3
"""Exact three-node policy algebra, independent of models and verl integration."""
import argparse
from fractions import Fraction as F
import itertools
import json
import math
from pathlib import Path


ACTIONS=tuple(itertools.product((0,1),repeat=3))


def probability(action, probs):
    return math.prod(p if a else 1-p for a,p in zip(action,probs,strict=True))


def analyze(probs, reward):
    probs=tuple(map(F,probs))
    if len(probs)!=3 or any(not 0<p<1 for p in probs):
        raise ValueError('three interior Bernoulli probabilities required')
    mass={a:probability(a,probs) for a in ACTIONS}
    values={a:F(reward(a)) for a in ACTIONS}
    mean=sum(mass[a]*values[a] for a in ACTIONS)
    conditional={}
    for j in range(3):
        for value in (0,1):
            support=[a for a in ACTIONS if a[j]==value]
            conditional[j,value]=sum(mass[a]*values[a] for a in support)/sum(mass[a] for a in support)-mean
    output=[]
    for j in range(3):
        score=lambda a:F(a[j])-probs[j]
        ordinary={a:(values[a]-mean)*score(a) for a in ACTIONS}
        marginalized={a:conditional[j,a[j]]*score(a) for a in ACTIONS}
        gradient=sum(mass[a]*ordinary[a] for a in ACTIONS)
        rb_gradient=sum(mass[a]*marginalized[a] for a in ACTIONS)
        variance=lambda xs:sum(mass[a]*(xs[a]-gradient)**2 for a in ACTIONS)
        assert gradient==rb_gradient
        assert variance(marginalized)<=variance(ordinary)
        output.append(dict(gradient=gradient,conditional_gradient=rb_gradient,
            centered_score_variance=variance(ordinary),conditional_score_variance=variance(marginalized),
            conditional_advantage_false=conditional[j,0],conditional_advantage_true=conditional[j,1]))
    return dict(expected_reward=mean,nodes=output)


def violation(action):
    i,s,b=action
    return int(b and not (i and s))


def run():
    p=(F(3,10),F(2,5),F(1,5))
    cases={
        'final_B_correct_only':analyze(p,lambda a:a[2]),
        'supervised_additive_gold111':analyze(p,lambda a:a[2]+F(1,10)*(a[0]+a[1])),
        'consistency_only':analyze(p,lambda a:1-violation(a)),
        'all_three_correct_gold111':analyze(p,lambda a:math.prod(a)),
    }
    # Frozen old conditional credits at a different current policy.
    new=(F(3,5),F(7,10),F(1,2))
    old_conditional=cases['all_three_correct_gold111']['nodes']
    marginal_gradient=[new[j]*(1-new[j])*(old_conditional[j]['conditional_advantage_true']-
        old_conditional[j]['conditional_advantage_false']) for j in range(3)]
    true_gradient=[new[j]*(1-new[j])*math.prod(new[k] for k in range(3) if k!=j) for j in range(3)]
    return dict(schema='jev-binding-node-credit-audit-v1',nodes=['I','S','B'],
        old_probabilities=p,cases=cases,
        consistency_truth_table=[dict(action=a,consistent=1-violation(a),gold111_correct=a==(1,1,1)) for a in ACTIONS],
        frozen_old_credit_away_from_old=dict(new_probabilities=new,
            reward='all_three_correct_gold111',marginal_unclipped_gradient=marginal_gradient,
            current_expected_reward_gradient=true_gradient,
            equivalent=False),
        method='Exact Fraction enumeration; independent Bernoulli logits; no clipping, GRPO normalization, CE, KL or neural parameters.',
        limitations=['Conditional credit preserves the expected on-policy score gradient and reduces each coordinate estimator variance versus the same centered-reward estimator.',
            'This does not prove smaller shared-parameter total gradient variance or higher model accuracy.',
            'Frozen old conditional advantages with marginal ratios generally cease to match interacting reward gradients away from old policy, even before clipping.',
            'Consistency has no gold correctness information; all-false is consistent.',
            'B implies I and S only when definition scopes align; reverse implication is not assumed.'],
        training_dispatch_allowed=False)


def serial(value):
    if isinstance(value,F):return dict(exact=str(value),value=float(value))
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with a.output.open('x') as f:json.dump(serial(run()),f,indent=2);f.write('\n')
