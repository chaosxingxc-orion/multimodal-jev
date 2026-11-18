#!/usr/bin/env python3
"""Exact binomial exploration diagnostics for complete candidate-set labels.

No fitted model probabilities. Exchangeable Bernoulli q is a controlled fixture,
not a claim about the task distribution or trained model. Missing labels exclude
set-level reward computation; they are not treated as negative candidates.
"""
import argparse
from collections import defaultdict
from fractions import Fraction
import json
import math
from pathlib import Path


def outcome_rewards(tp, fp, fn, tn):
    n=tp+fp+fn+tn
    if (any(type(x) is not int or x<0 for x in (tp,fp,fn,tn)) or n==0):
        raise ValueError('complete nonnegative confusion counts required')
    denominator=2*tp+fp+fn
    return dict(bit_accuracy=Fraction(tp+tn,n),exact_set=Fraction(int(fp==fn==0)),
        set_f1=Fraction(2*tp,denominator) if denominator else Fraction(1))


def exact_diagnostic(candidates, positives, q, group_size=16):
    if (type(candidates) is not int or type(positives) is not int or not 0<=positives<=candidates or
            candidates<1 or type(q) not in (float,int) or not 0<=q<=1 or
            type(group_size) is not int or group_size<2):raise ValueError('bounded fixture required')
    mass={name:defaultdict(float) for name in ('bit_accuracy','exact_set','set_f1')}
    for tp in range(positives+1):
        pos_mass=math.comb(positives,tp)*q**tp*(1-q)**(positives-tp)
        for fp in range(candidates-positives+1):
            probability=pos_mass*math.comb(candidates-positives,fp)*q**fp*(1-q)**(candidates-positives-fp)
            if not probability:continue
            for name,reward in outcome_rewards(tp,fp,positives-tp,candidates-positives-fp).items():
                mass[name][reward]+=probability
    result={}
    for name,distribution in mass.items():
        if abs(sum(distribution.values())-1)>1e-12:raise ValueError('probability mass lost')
        mean=sum(float(r)*p for r,p in distribution.items())
        result[name]=dict(mean=mean,variance=sum((float(r)-mean)**2*p for r,p in distribution.items()),
            constant_reward_group_probability=sum(p**group_size for p in distribution.values()),
            zero_reward_probability=distribution.get(Fraction(0),0.))
    controls={}
    for name,tp,fp in [('return_none',0,0),('return_all',positives,candidates-positives),
                       ('oracle_single',int(positives>0),0),('oracle_full',positives,0)]:
        controls[name]={k:float(v) for k,v in outcome_rewards(tp,fp,positives-tp,candidates-positives-fp).items()}
    return dict(candidates=candidates,positives=positives,sampling_probability=q,group_size=group_size,
                rewards=result,controls=controls)


def report():
    cases=[exact_diagnostic(n,k,q) for n,k in [(60,1),(6,1),(6,2),(6,0)] for q in (.5,.1)]
    return dict(cases=cases,training_dispatch_allowed=False,production_reward=None,
        empty_set_convention='set_f1(empty_prediction,empty_gold)=1; nonempty prediction against empty gold=0',
        partial_labels='no complete-set reward without complete set truth',
        caveats=['Bernoulli actions are sampling policy, not independent semantic truth.',
                 'Reward F1 encourages a decision set, not calibrated marginal probabilities.',
                 'Exact-set on empty gold can favor always-empty when many queries have empty truth.',
                 'Reward scale/variance interact with GRPO standardization; no weights or algorithm change adopted.'],
        scope='Exact controlled exploration arithmetic and degenerate controls; no empirical model or RL improvement.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with a.output.open('x') as f:json.dump(report(),f,indent=2);f.write('\n')
