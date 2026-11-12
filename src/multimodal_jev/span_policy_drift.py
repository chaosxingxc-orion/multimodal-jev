"""Post-update finite-policy drift, distinct from same-policy pre-update clipping.

Re-normalization removes input log-softmax rounding in diagnostics only. This
module never changes training probabilities, objective, or acceptance thresholds.
"""
import math


def distribution(logp):
    if len(logp)<2 or any(type(x) not in (float,int) or not math.isfinite(x) or x>1e-6 for x in logp):
        raise ValueError('finite categorical log probabilities required')
    mass=sum(math.exp(x) for x in logp)
    if mass<=0 or abs(math.log(mass))>1e-6:raise ValueError('normalized full candidate policy required')
    return [x-math.log(mass) for x in logp]


def diagnose(old,new,target,clip=.2):
    if type(target)!=int or not 0<=target<len(old) or len(old)!=len(new) or not 0<clip<1:
        raise ValueError('aligned target/support and bounded clipping required')
    a,b=distribution(old),distribution(new);p,q=[list(map(math.exp,x)) for x in (a,b)]
    delta=[y-x for x,y in zip(a,b,strict=True)]
    low,high=math.log1p(-clip),math.log1p(clip)
    # Exact old-policy mass in clipped regions for correctness-centered advantage.
    outside=sum(prob for prob,d in zip(p,delta) if d<low or d>high)
    clipped_mass=sum(prob for i,(prob,d) in enumerate(zip(p,delta))
                     if (i==target and d>high) or (i!=target and d<low))
    surrogate=0.; clipped=0.; dual=0.
    for i,(prob,d) in enumerate(zip(p,delta)):
        advantage=(i==target)-p[target]
        ratio=math.exp(max(-20.,min(20.,d)))  # Same numeric clamp as training reference.
        term=advantage*ratio
        bounded=advantage*min(1+clip,max(1-clip,ratio))
        value=min(term,bounded)
        surrogate+=prob*term;clipped+=prob*value
        dual+=prob*(max(3*advantage,value) if advantage<0 else value)
    return dict(kl_old_to_new=max(0.,sum(x*(la-lb) for x,la,lb in zip(p,a,b))),
        kl_new_to_old=max(0.,sum(x*(lb-la) for x,la,lb in zip(q,a,b))),
        total_variation=.5*sum(abs(x-y) for x,y in zip(p,q)),
        maximum_abs_log_ratio=max(map(abs,delta)),old_mass_outside_clip=outside,
        old_mass_clipped_by_advantage=clipped_mass,
        postupdate_unclipped_surrogate=surrogate,postupdate_clipped_surrogate=clipped,
        postupdate_dual_clipped_surrogate=dual,
        gold_probability_before=p[target],gold_probability_after=q[target],
        gold_probability_delta=q[target]-p[target],
        argmax_changed=max(range(len(a)),key=a.__getitem__)!=max(range(len(b)),key=b.__getitem__))
