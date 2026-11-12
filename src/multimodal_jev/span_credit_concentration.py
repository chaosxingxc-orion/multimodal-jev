"""On-policy output-logit credit geometry for finite additive rewards.

This is independent-logit sensitivity, not shared-parameter gradient attribution
and not an alternative training objective. Candidate mass is normalized only
for this analytic diagnostic to remove FP32 log-softmax rounding residuals.
"""
import math


def sensitivity(logprobs, target, weight=1.):
    if (len(logprobs) < 2 or type(target) is not int or not 0 <= target < len(logprobs)
            or not math.isfinite(weight) or weight < 0
            or any(not math.isfinite(p) or p > 1e-6 for p in logprobs)):
        raise ValueError('finite complete categorical support and target required')
    mass = sum(math.exp(p) for p in logprobs)
    if mass <= 0 or abs(math.log(mass)) > 1e-6:
        raise ValueError('normalized policy required')
    probability = [math.exp(p)/mass for p in logprobs]; py = probability[target]
    gradient = [weight*py*((i == target)-p) for i, p in enumerate(probability)]
    return dict(gold_probability=py, gradient=gradient,
                norm=math.sqrt(sum(g*g for g in gradient)), input_mass_error=abs(mass-1))


def concentration(rows):
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('distinct nonempty request population required')
    if any(not math.isfinite(r['norm']) or r['norm'] < 0 for r in rows):
        raise ValueError('nonnegative finite norms required')
    total = sum(r['norm'] for r in rows)
    if total == 0:
        return dict(requests=len(rows), sum_norms=0., effective_requests=0., top2_share=None,
                    top10percent_share=None, below_one_thousandth_share=None)
    ordered = sorted(rows, key=lambda r: (-r['norm'], r['id']))
    shares = [r['norm']/total for r in ordered]
    return dict(requests=len(rows), sum_norms=total, effective_requests=1/sum(s*s for s in shares),
        top2_share=sum(shares[:2]), top10percent_share=sum(shares[:math.ceil(len(rows)/10)]),
        below_one_thousandth_share=sum(s for s, r in zip(shares, ordered) if r['gold_probability'] < .001))
