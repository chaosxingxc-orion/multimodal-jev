"""Finite categorical policy parity, including worst-case complete action ratio."""
import math
import torch


def compare_span_policies(actor, behavior, per_span_tolerance=0.02, joint_ratio_bounds=(0.98,1.02)):
    """One request, ragged candidate distributions; no sampling hides rare errors.

    For independent finite actions, extrema of joint log ratio are sums of
    per-span extrema. This checks every possible tuple without enumerating it.
    Inputs are detached CPU diagnostics, not an off-policy correction.
    """
    if len(actor)!=len(behavior) or not actor:
        raise ValueError('aligned nonempty Span policies required')
    low,high=joint_ratio_bounds
    if not 0<low<=1<=high or not math.isfinite(high) or not math.isfinite(per_span_tolerance) or per_span_tolerance<0:
        raise ValueError('valid finite parity bounds required')
    errors=[];lower=upper=0.;kl=0.;flips=[]
    for i,(a,b) in enumerate(zip(actor,behavior)):
        a=torch.as_tensor(a,dtype=torch.float64).detach().cpu()
        b=torch.as_tensor(b,dtype=torch.float64).detach().cpu()
        if a.ndim!=1 or a.shape!=b.shape or len(a)<2 or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('finite aligned candidate log probabilities required')
        if abs(float(a.logsumexp(0)))>1e-6 or abs(float(b.logsumexp(0)))>1e-6:
            raise ValueError('normalized candidate policies required')
        delta=a-b
        errors.extend(delta.abs().tolist())
        lower+=float(delta.min());upper+=float(delta.max())
        kl+=float((a.exp()*delta).sum())
        if int(a.argmax())!=int(b.argmax()):flips.append(i)
    per_pass=max(errors)<=per_span_tolerance
    joint_pass=math.log(low)<=lower and upper<=math.log(high)
    return dict(passed=per_pass and joint_pass,per_span_passed=per_pass,joint_ratio_passed=joint_pass,
                maximum_logprob_error=max(errors),p95_logprob_error=float(torch.tensor(errors).quantile(.95)),
                joint_log_ratio_min=lower,joint_log_ratio_max=upper,
                joint_ratio_min=math.exp(lower) if lower>-745 else 0.,
                joint_ratio_max=math.exp(upper) if upper<709 else None,
                candidate_kl_actor_to_behavior=max(0.,kl),argmax_changed_spans=flips,
                per_span_tolerance=per_span_tolerance,joint_ratio_bounds=list(joint_ratio_bounds),
                scope='Every finite complete action; fixed independent categorical sampling, temperature already applied.')
