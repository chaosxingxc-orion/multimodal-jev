"""Supervised candidate InfoNCE for the simple exclusive-event task.

The model still emits independent sigmoid probabilities. The softmax here is
only the contrastive training objective; it is not the inference readout.
"""
import math

import torch


def loss(logits, labels, observed, request_weights, real_requests, *,
         global_real_requests, world_size=1, temperature=1., probability_margin=None):
    if (logits.ndim != 2 or min(logits.shape) < 1 or not logits.is_floating_point()
            or not torch.isfinite(logits).all()
            or labels.dtype != torch.bool or observed.dtype != torch.bool
            or labels.shape != logits.shape or observed.shape != logits.shape
            or labels.device != logits.device or observed.device != logits.device):
        raise ValueError('finite candidate scores and matching Boolean supervision required')
    if probability_margin is not None:
        validate_margin(probability_margin)
    b = logits.shape[0]
    if (request_weights.shape != (b,) or not request_weights.is_floating_point()
            or request_weights.requires_grad or not torch.isfinite(request_weights).all()
            or real_requests.shape != (b,) or real_requests.dtype != torch.bool
            or request_weights.device != logits.device or real_requests.device != logits.device
            or not torch.equal(request_weights, real_requests.to(request_weights.dtype))):
        raise ValueError('uniform real queries and zero-weight padding required')
    if (type(global_real_requests) is not int or global_real_requests < 1
            or int(real_requests.sum()) > global_real_requests
            or type(world_size) is not int or world_size < 1
            or type(temperature) not in (float, int)
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('global real denominator, DP width and positive temperature required')
    # Only real rows require supervision. Padding has no effect even on a rank
    # whose entire tail consists of padding; unknown labels never become false.
    active = real_requests
    if ((~observed[active]).any() or logits.shape[1] < 2
            or (labels[active].sum(-1) != 1).any()):
        raise ValueError('complete exclusive-event panel with one positive and negatives required')
    z = logits.float() if logits.dtype in (torch.bfloat16, torch.float16) else logits
    if active.any():
        scores = z[active] / temperature
        positive = scores[labels[active]]
        numerator = (torch.logsumexp(scores, dim=-1) - positive).sum()
    else:
        numerator = z.sum() * 0
    metrics = {}
    total = numerator
    if probability_margin is not None:
        c = probability_margin
        if active.any():
            scores = z[active]
            positive = scores[labels[active]]
            negative = scores.masked_fill(labels[active], -torch.inf).amax(-1)
            ppos, pneg = positive.sigmoid(), negative.sigmoid()
            gap = ppos - pneg
            gap_loss = (c['margin'] - gap).clamp_min(0).sum()
            # Anchors in logit space avoid vanishing corrective gradients when
            # the sigmoid has saturated on the wrong side of the threshold.
            hi = c['decision_threshold'] + c['margin']/2
            lo = c['decision_threshold'] - c['margin']/2
            pos_anchor = (math.log(hi/(1-hi)) - positive).clamp_min(0)
            neg_anchor = (negative - math.log(lo/(1-lo))).clamp_min(0)
            anchor = (pos_anchor + neg_anchor).sum()/2
            metrics = dict(probability_gap_sum=float(gap.detach().sum()),
                margin_satisfied_count=int((gap >= c['margin']).sum()),
                anchored_count=int(((ppos >= hi) & (pneg <= lo)).sum()))
        else:
            gap_loss = anchor = z.sum()*0
            metrics = dict(probability_gap_sum=0.,margin_satisfied_count=0,anchored_count=0)
        total = numerator + c['gap_weight']*gap_loss + c['anchor_weight']*anchor
        metrics.update(probability_margin_numerator=float(gap_loss.detach()),
            threshold_anchor_numerator=float(anchor.detach()),
            total_numerator=float(total.detach()),probability_margin=c)
    value = total * world_size / global_real_requests
    return value, dict(infonce_numerator=float(numerator.detach()),
        local_real_queries=int(active.sum()), global_real_queries=global_real_requests,
        temperature=temperature, calibration_claim=False, **metrics)


def validate_margin(c):
    if not isinstance(c, dict) or set(c) != {'margin','decision_threshold','gap_weight','anchor_weight'}:
        raise ValueError('complete explicit probability margin configuration required')
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in c.values()):
        raise ValueError('finite margin parameters required')
    if not (0 < c['margin'] < 1 and c['margin']/2 < c['decision_threshold'] < 1-c['margin']/2
            and c['gap_weight'] > 0 and c['anchor_weight'] > 0):
        raise ValueError('feasible margin anchors and positive loss weights required')


def validate_objective(c):
    if c.get('kind') == 'candidate_infonce':
        expected = {'kind','temperature','probability_coefficient'}
    elif c.get('kind') in ('candidate_infonce_probability_margin_v1', 'intent_triplet_infonce_v1'):
        expected = {'kind','temperature','probability_coefficient','probability_margin'}
        if c['kind'] == 'intent_triplet_infonce_v1' and 'negative_cards_per_anchor' in c:
            expected.add('negative_cards_per_anchor')
            if type(c['negative_cards_per_anchor']) is not int or c['negative_cards_per_anchor'] < 1:
                raise ValueError('positive integer negative_cards_per_anchor required')
        validate_margin(c.get('probability_margin'))
    else:
        raise ValueError('unknown Stage1 objective')
    if (set(c) != expected or c['probability_coefficient'] != 0.
            or type(c['temperature']) not in (int,float)
            or not math.isfinite(c['temperature']) or c['temperature'] <= 0):
        raise ValueError('explicit contrastive Stage1 objective without binary CE required')
