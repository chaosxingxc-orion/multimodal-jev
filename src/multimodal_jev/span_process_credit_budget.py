"""Exact independent-logit geometry for additive process/Final correctness.

Disjoint output-logit coordinates do not measure shared parameter gradients.
All weights retain their original scale; no per-Span standardization occurs.
"""
import math
from .span_credit_concentration import sensitivity


def measure(logprobs,targets,weights):
    if (len(logprobs)<2 or len(logprobs)!=len(targets) or len(weights)!=len(targets)
            or any(type(w) not in (int,float) or not math.isfinite(w) or w<0 for w in weights)
            or weights[-1]<=0):
        raise ValueError('nonnegative additive process weights and positive Final required')
    spans=[sensitivity(p,y,w) for p,y,w in zip(logprobs,targets,weights,strict=True)]
    process_norm=math.sqrt(sum(s['norm']**2 for s in spans[:-1]));final_norm=spans[-1]['norm']
    total_squared=process_norm**2+final_norm**2
    return dict(gold_probabilities=[s['gold_probability'] for s in spans],
        span_norms=[s['norm'] for s in spans],process_norm=process_norm,final_norm=final_norm,
        process_to_final_norm=process_norm/final_norm if final_norm>0 else None,
        process_squared_norm_share=process_norm**2/total_squared if total_squared>0 else None,
        process_exceeds_final=process_norm>final_norm,
        expected_process_reward=sum(w*s['gold_probability'] for w,s in zip(weights[:-1],spans[:-1],strict=True)),
        expected_final_reward=weights[-1]*spans[-1]['gold_probability'])
