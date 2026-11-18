"""Candidate contrastive ranking plus optional proper-scoring supervision.

Independent candidate logits, no softmax across candidates. Pairwise ranking
compares every known positive with every known negative, never two positives.
The caller supplies a single global real-query denominator across DP/microsteps.
Missing candidates have zero gradient; missing queries remain in the population
denominator. This is auxiliary supervision, not a replacement for GRPO/RLOO.
"""
import math
import torch
from torch.nn import functional as F


def loss(logits, labels, observed, request_weights, real_requests, *,
         global_real_requests, world_size=1, ranking_coefficient, probability_coefficient,
         temperature=1., margin=0., probability_score='binary_nll'):
    if (logits.ndim != 2 or min(logits.shape) < 1 or not logits.is_floating_point() or
            not torch.isfinite(logits).all() or labels.dtype != torch.bool or
            observed.dtype != torch.bool or labels.shape != logits.shape or observed.shape != logits.shape):
        raise ValueError('finite B x candidates logits, Boolean truth and observation mask required')
    b, _ = logits.shape
    if (request_weights.shape != (b,) or not request_weights.is_floating_point() or
            request_weights.requires_grad or not torch.isfinite(request_weights).all() or
            (request_weights < 0).any() or real_requests.shape != (b,) or real_requests.dtype != torch.bool or
            any(t.device != logits.device for t in (labels, observed, request_weights, real_requests)) or
            (request_weights[~real_requests] != 0).any()):
        raise ValueError('detached query weights and zero-weight padding required')
    if (type(global_real_requests) is not int or global_real_requests < 1 or
            int(real_requests.sum()) > global_real_requests or type(world_size) is not int or world_size < 1):
        raise ValueError('authenticated global real query denominator required')
    if (any(type(v) not in (int,float) or not math.isfinite(v) or v < 0
            for v in (ranking_coefficient, probability_coefficient, margin)) or
            type(temperature) not in (int,float) or not math.isfinite(temperature) or temperature <= 0 or
            probability_score not in ('binary_nll','brier')):
        raise ValueError('explicit finite nonnegative coefficients, positive temperature and proper score required')
    z = logits.float() if logits.dtype in (torch.float16,torch.bfloat16) else logits
    ranking = z.new_zeros(())
    probability = z.new_zeros(())
    pair_counts = []; known_counts = []
    for i in range(b):
        known = observed[i]
        positives = z[i,known & labels[i]]
        negatives = z[i,known & ~labels[i]]
        count = int(known.sum()); pairs = positives.numel()*negatives.numel()
        known_counts.append(count); pair_counts.append(pairs)
        # Empty terms retain a valid zero gradient, including all-unlabeled rows.
        pair = z[i].sum()*0
        proper = z[i].sum()*0
        if pairs:
            pair = F.softplus((negatives[None,:]-positives[:,None]+margin)/temperature).mean()
        if count:
            values, truth = z[i,known], labels[i,known].to(z.dtype)
            proper = (F.binary_cross_entropy_with_logits(values,truth,reduction='mean')
                      if probability_score == 'binary_nll' else (values.sigmoid()-truth).square().mean())
        ranking = ranking+request_weights[i]*pair
        probability = probability+request_weights[i]*proper
    scale = world_size/global_real_requests
    total = scale*(ranking_coefficient*ranking+probability_coefficient*probability)
    return total, dict(ranking_numerator=float(ranking.detach()),probability_numerator=float(probability.detach()),
        observed_counts=known_counts,positive_negative_pair_counts=pair_counts,
        local_real_requests=int(real_requests.sum()),global_real_requests=global_real_requests,
        normalization='weighted_sum_per_query_mean_terms_divided_by_global_real_queries',
        scope='Auxiliary supervised objective only; no reward, sampling, calibration or RL-gain claim.')
