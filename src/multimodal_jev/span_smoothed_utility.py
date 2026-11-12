"""Candidate bounded log-probability utility for one reviewed Final Span.

U(p)=log((p+tau)/(1+tau))/log((1+tau)/tau), tau=1/K².
U lies in [-1,0]. Freeze U'(p_old) as the correctness reward scale during
one policy update. Its on-policy gradient equals grad U; clipped off-policy
PPO is only a local surrogate. No group/batch normalization of these scales.
"""
import math


def utility_credit(probability, support):
    if type(support) is not int or support<2:
        raise ValueError('categorical support of at least two required')
    if type(probability) not in (int,float) or not math.isfinite(probability) or not 0<=probability<=1:
        raise ValueError('finite gold probability in unit interval required')
    tau=1/(support*support);normalizer=math.log1p(1/tau)
    return dict(utility=math.log((probability+tau)/(1+tau))/normalizer,
                scale=1/((probability+tau)*normalizer),tau=tau,
                maximum_scale=1/(tau*normalizer))


def transform(items):
    """Scale per-request Span credit before global DP transport; preserve weights.

Inputs and old probabilities remain identical. Final labels stay outside model
inputs. This restricted candidate never scales multi-Span process rewards.
"""
    import torch
    from .span_finite_credit import validate_policy
    if not items:raise ValueError('nonempty full batch required')
    result=[];receipts=[]
    for item in items:
        old=item['behavior'];validate_policy(old,1,detached=True)
        y,mask,w=item['targets'],item['supervised'],item['span_weights']
        if (len(old)!=1 or y.shape!=(1,1) or y.dtype!=torch.long or mask.shape!=(1,1)
                or mask.dtype!=torch.bool or w.shape!=(1,1) or w.requires_grad
                or not w.is_floating_point() or not torch.isfinite(w).all() or w.item()!=1.):
            raise ValueError('single Final with original unit credit required')
        if bool(mask.item()):
            target=y.item();p=old[0]
            if not 0<=target<p.shape[1]:raise ValueError('reviewed target outside support')
            measured=utility_credit(p[0,target].exp().item(),p.shape[1])
            scaled=w*measured['scale']
        else:
            measured=None;scaled=w.clone()
        result.append(dict(item,span_weights=scaled.detach()))
        receipts.append(dict(request_id=item['request_id'],supervised=bool(mask.item()),credit=measured))
    return result,dict(schema='jev-final-smoothed-utility-v1',records=receipts,
        normalization='original effective request weights, unchanged global denominator',
        scope='Candidate single-Span locally utility-matched PPO credit; not original mean correctness objective or process-credit policy.')
