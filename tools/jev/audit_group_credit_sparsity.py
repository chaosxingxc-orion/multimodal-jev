#!/usr/bin/env python3
"""Enumerate multinomial group counts for Final and one independent process Q."""
import argparse
import itertools
import json
import math
from pathlib import Path


def group_terms(counts, final_probability, process_probability, process_weight):
    """Return group-mean score gradients at old=current, without CE/KL/clipping."""
    actions=((0,0),(0,1),(1,0),(1,1))  # (Final, Q)
    g=sum(counts)
    if g<2 or len(counts)!=4 or any(type(n) is not int or n<0 for n in counts):
        raise ValueError('four nonnegative counts and group size >=2 required')
    rewards=[f+process_weight*q for f,q in actions]
    mean=sum(n*r for n,r in zip(counts,rewards))/g
    variance=sum(n*(r-mean)**2 for n,r in zip(counts,rewards))/(g-1)
    centered=[r-mean for r in rewards]
    scores=[(f-final_probability,q-process_probability) for f,q in actions]
    gradients={}
    for name,scale in [('grpo',1/(math.sqrt(variance)+1e-6)),('rloo',g/(g-1))]:
        gradients[name]=[sum(n*a*s[j] for n,a,s in zip(counts,centered,scores))*scale/g for j in (0,1)]
    final_correct=counts[2]+counts[3]
    return dict(gradients=gradients, final_all_wrong=final_correct==0,
        final_constant=final_correct in (0,g), total_reward_constant=variance<1e-28)


def enumerate_case(pf,pq,weight,g=16):
    if not 0<pf<1 or not 0<pq<1 or weight<0 or g<2:
        raise ValueError('interior probabilities, nonnegative weight and group size >=2 required')
    probabilities=((1-pf)*(1-pq),(1-pf)*pq,pf*(1-pq),pf*pq)
    totals={name:[0.,0.] for name in ('grpo','rloo')}
    constant_contrib={name:[0.,0.] for name in totals}
    mass=all_wrong=constant=reward_constant=0.;groups=0
    for a in range(g+1):
        for b in range(g-a+1):
            for c in range(g-a-b+1):
                counts=(a,b,c,g-a-b-c)
                coefficient=math.comb(g,a)*math.comb(g-a,b)*math.comb(g-a-b,c)
                probability=coefficient*math.prod(p**n for p,n in zip(probabilities,counts))
                terms=group_terms(counts,pf,pq,weight)
                mass+=probability;groups+=1
                all_wrong+=probability*terms['final_all_wrong']
                constant+=probability*terms['final_constant']
                reward_constant+=probability*terms['total_reward_constant']
                for name in totals:
                    for j in (0,1):
                        v=probability*terms['gradients'][name][j]
                        totals[name][j]+=v
                        if terms['final_constant']:constant_contrib[name][j]+=v
    return dict(final_probability=pf,process_probability=pq,process_reward_weight=weight,group_size=g,
        enumerated_count_vectors=groups,probability_mass=mass,final_all_wrong_probability=all_wrong,
        final_constant_probability=constant,total_reward_constant_probability=reward_constant,
        exact_expected_reward_gradient=[pf*(1-pf),weight*pq*(1-pq)],
        expected_group_logit_gradients=totals,
        final_constant_groups_unconditional_gradient_contribution=constant_contrib,
        grpo_process_gradient_fraction_from_final_constant_groups=(constant_contrib['grpo'][1]/totals['grpo'][1] if weight else None))


def audit():
    cases=[enumerate_case(p,.5,w,g) for g in (16,32) for p in (1/60,.1,.5,.9) for w in (0.,.025,.1)]
    return dict(schema='jev-group-credit-sparsity-audit-v1',cases=cases,
        method='Exhaustive multinomial count-vector enumeration, floating-point arithmetic, no Monte Carlo; G+3 choose 3 histograms cover all ordered groups.',
        model='One 60-way Final collapsed into gold vs all others (gold-logit score f-p), plus one independent binary process correctness (score q-p).',
        reward='Final correctness plus one process term; .025 is one labelled Q in a configured 4I+4S panel; .1 is one single-Q family. Other rewards are absent, not redistributed.',
        limits=['Correctness probabilities are hypotheses, not measured model accuracy or Stage1 outputs.',
            'One stochastic process Q cannot represent all correlations and variances of a fully supervised eight-Q panel.',
            'Old=current point; no off-policy clipping, shared-backbone Jacobians, CE, KL or optimizer.',
            'Independent-coordinate gradient magnitudes are not shared-parameter gradient budgets or performance gains.'],
        training_dispatch_allowed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    r=audit();a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({'cases':len(r['cases']),'output':str(a.output)}))
