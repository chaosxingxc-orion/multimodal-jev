"""Exact conditional compatibility credit for factorized categorical spans.

Candidate exploratory penalty -mu * incompatible. Not correctness reward:
consistent wrong triples receive no penalty. No sampled/joint-PPO claim.
"""
import math


def credit(logprobs, pairs, weight=.1):
    if (len(logprobs)!=3 or type(weight) not in (int,float)
            or not math.isfinite(weight) or not 0<=weight<1):
        raise ValueError('three finite policies and bounded relationship weight required')
    probabilities=[]
    for values in logprobs:
        if len(values)<2 or any(type(p) not in (int,float) or not math.isfinite(p) or p>1e-6 for p in values):
            raise ValueError('finite complete categorical support required')
        p=[math.exp(v) for v in values];mass=sum(p)
        if abs(math.log(mass))>1e-6:raise ValueError('normalized policy required')
        probabilities.append([v/mass for v in p])
    s,a,f=probabilities
    if len(pairs)!=len(f) or any(not isinstance(x,(tuple,list)) or len(x)!=2 for x in pairs):
        raise ValueError('one canonical pair per Final candidate required')
    if (any(type(i) is not int or type(j) is not int or not 0<=i<len(s) or not 0<=j<len(a) for i,j in pairs)
            or len({tuple(x) for x in pairs})!=len(pairs)):
        raise ValueError('distinct in-support canonical pairs required')
    conditional=[[0.]*len(p) for p in probabilities]
    expected=0.
    for k,(i,j) in enumerate(pairs):
        conditional[0][i]+=a[j]*f[k]
        conditional[1][j]+=s[i]*f[k]
        conditional[2][k]=s[i]*a[j]
        expected+=s[i]*a[j]*f[k]
    advantages=[[weight*(c-expected) for c in values] for values in conditional]
    # Derivative of expected reward w.r.t. each independent logit vector.
    gradients=[[p*v for p,v in zip(ps,adv,strict=True)] for ps,adv in zip(probabilities,advantages,strict=True)]
    return dict(expected_compatibility=expected,expected_penalty=-weight*(1-expected),
                conditional_compatibility=conditional,advantages=advantages,logit_gradients=gradients,
                scope='Factorized conditional relationship credit only. Compatibility is not correctness; no shared parameter gradient or efficacy claim.')
