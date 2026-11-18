#!/usr/bin/env python3
"""Enumerate finite product policies to audit clipping geometry, not model efficacy."""
import argparse
from fractions import Fraction as F
import itertools
import json
import math
from pathlib import Path


def audit_policy(old, new, weights, clip=F(1, 5), clipped=True):
    """Exact reward=weighted binary correctness; analytic independent-logit gradients.

    Advantages are old-policy centered reward (not sampled GRPO advantages).
    Old probabilities and advantages are held fixed. Return maximization objective.
    """
    old, new, weights = tuple(map(F, old)), tuple(map(F, new)), tuple(map(F, weights))
    if not old or not len(old) == len(new) == len(weights):
        raise ValueError('aligned nonempty vectors required')
    if any(not 0 < p < 1 for p in old+new) or any(w < 0 for w in weights):
        raise ValueError('interior probabilities and nonnegative weights required')
    if not 0 < clip < 1:
        raise ValueError('bounded clip required')
    baseline = sum(p*w for p,w in zip(old, weights))
    objective = F(0); outside = F(0); saturated = F(0); kinks = F(0)
    gradients = [F(0)]*len(old)
    minimum, maximum = None, None
    for action in itertools.product((0,1), repeat=len(old)):
        pold = math.prod(p if a else 1-p for a,p in zip(action,old))
        pnew = math.prod(p if a else 1-p for a,p in zip(action,new))
        ratio = pnew/pold
        minimum = ratio if minimum is None else min(minimum,ratio)
        maximum = ratio if maximum is None else max(maximum,ratio)
        adv = sum(a*w for a,w in zip(action,weights))-baseline
        outside += pold*(ratio < 1-clip or ratio > 1+clip)
        inactive = clipped and ((adv > 0 and ratio > 1+clip) or (adv < 0 and ratio < 1-clip))
        kink = clipped and ((adv > 0 and ratio == 1+clip) or (adv < 0 and ratio == 1-clip))
        kinks += pold*kink
        saturated += pold*inactive
        effective = max(1-clip,min(1+clip,ratio)) if inactive else ratio
        objective += pold*effective*adv
        if not inactive:
            for j in range(len(old)):
                gradients[j] += pold*ratio*adv*(action[j]-new[j])
    return dict(objective=objective, logit_gradient=gradients,
                ratio_outside_old_mass=outside, surrogate_saturated_old_mass=saturated,
                nondifferentiable_old_mass=kinks, ratio_min=minimum, ratio_max=maximum)


def marginalized_final(old, new, final_weight=F(1)):
    # Valid here only because reward has no auxiliary-action dependence and policy factorizes.
    return audit_policy([old[-1]], [new[-1]], [final_weight])


def serialize(result):
    return {k: [float(v) for v in x] if isinstance(x,list) else float(x) for k,x in result.items()}


def run_audit():
    records=[]
    for final_new in (F(1,5),F(11,50)):
        for n in (0,2,8):
            old=[F(1,2)]*n+[F(1,5)]
            new=[F(11,20)]*n+[final_new]
            weights=[F(0)]*n+[F(1)]
            joint=audit_policy(old,new,weights)
            unclipped=audit_policy(old,new,weights,clipped=False)
            marginal=marginalized_final(old,new)
            assert all(g==0 for g in unclipped['logit_gradient'][:-1])
            assert unclipped['logit_gradient'][-1]==final_new*(1-final_new)
            assert joint['nondifferentiable_old_mass']==0
            kl=[float(p)*math.log(float(p/q))+float(1-p)*math.log(float((1-p)/(1-q)))
                for p,q in zip(new,old)]
            records.append(dict(auxiliary_questions=n,old_auxiliary_probability=.5,
                new_auxiliary_probability=.55,old_final_probability=.2,new_final_probability=float(final_new),
                reward='Final correctness only; every auxiliary reward weight is zero',
                exact_joint_clipped=serialize(joint),exact_unclipped=serialize(unclipped),
                exact_reward_marginalized_clipped=serialize(marginal),
                auxiliary_gradient_sum=float(sum(joint['logit_gradient'][:-1])),
                kl_new_to_old=dict(final=kl[-1],auxiliary_sum=sum(kl[:-1]),joint_sum=sum(kl))))
    return dict(schema='jev-joint-span-clipping-audit-v1',cases=records,
        method='Exact Fraction enumeration; 1, 3 or 9 independent Bernoulli outputs. Gradient is with respect to independent current logits; fixed old probabilities and exact centered reward.',
        scope='Policy geometry diagnostic, not sampled GRPO/RLOO implementation, shared-backbone gradient, optimizer update, NPU acceptance or task performance.',
        implications=['Reward masking does not remove an action coordinate from the joint likelihood ratio.',
            'Auxiliary drift can change clipped Final gradient and introduce auxiliary surrogate gradients despite Final-only reward.',
            'Marginalizing irrelevant actions removes this effect in this factorized Final-only case; it is a separate objective variant.',
            'Exact joint KL is the sum of marginal KL; constant coefficient across changing Q counts changes the aggregate regularization geometry.',
            'At the old-policy point unclipped score expectations are correct; clipping away from that point is the issue.'],
        training_dispatch_allowed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=run_audit();a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
