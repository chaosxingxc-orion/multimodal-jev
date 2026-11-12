"""Prospective batch-relative weighting diagnostic, not a production reward.

For reviewed single-Span classification: c_i=1/max(p_old(y_i),1/K_i),
normalized by the original weighted batch mean. All factors are detached when
used in a policy objective. This changes the objective's source weighting.
"""
import math


def bounded_scales(gold_probabilities, supports, request_weights):
    n=len(gold_probabilities)
    if not n or len(supports)!=n or len(request_weights)!=n:
        raise ValueError('aligned nonempty population required')
    if any(type(k) is not int or k<2 for k in supports):raise ValueError('finite categorical supports required')
    if any(type(p) not in (int,float) or not math.isfinite(p) or not 0<=p<=1 for p in gold_probabilities):
        raise ValueError('finite gold probabilities required')
    if any(type(w) not in (int,float) or not math.isfinite(w) or w<0 for w in request_weights) or sum(request_weights)<=0:
        raise ValueError('positive effective request weight required')
    raw=[1/max(p,1/k) for p,k in zip(gold_probabilities,supports,strict=True)]
    mean=sum(w*c for w,c in zip(request_weights,raw,strict=True))/sum(request_weights)
    return dict(raw_scales=raw, weighted_mean_raw_scale=mean, scales=[c/mean for c in raw],
                floor='1 / candidate count', normalization='original request-weighted batch mean')
