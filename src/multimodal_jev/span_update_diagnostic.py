"""Descriptive same-batch changes; never an acceptance or selection metric."""
import torch

from .span_credit_diagnostic import exact_additive_logit_gradient
from .span_rule_objectives import expected_rule_reward


def diagnose_update(before, after, targets, supervised, auxiliary_weight):
    if len(before) != len(after):
        raise ValueError('aligned policies required')
    # These calls validate normalized finite policies and supervision.
    gradient = exact_additive_logit_gradient(before, targets, supervised, auxiliary_weight)
    exact_additive_logit_gradient(after, targets, supervised, auxiliary_weight)
    old_reward = expected_rule_reward(before, targets, supervised, auxiliary_weight)
    new_reward = expected_rule_reward(after, targets, supervised, auxiliary_weight)
    rows = []
    for j, (old, new) in enumerate(zip(before, after, strict=True)):
        mask = supervised[:, j]
        gold = targets[:, j].masked_fill(~mask, 0)
        p_old = old.exp().gather(1, gold[:, None]).squeeze(1)
        p_new = new.exp().gather(1, gold[:, None]).squeeze(1)
        # log probabilities are a valid softmax logit coordinate system.
        # Gradient sums to zero, making this invariant to row-wise logit offsets.
        linear = (gradient[:, j] * (new - old)).sum(-1)
        def summary(active):
            count = int(active.sum())
            if not count:
                return dict(requests=0)
            return dict(requests=count,
                gold_probability_before=float(p_old[active].mean()),
                gold_probability_after=float(p_new[active].mean()),
                gold_probability_delta=float((p_new-p_old)[active].mean()),
                accuracy_before=float((old.argmax(-1)[active] == gold[active]).float().mean()),
                accuracy_after=float((new.argmax(-1)[active] == gold[active]).float().mean()),
                predicted_true_before=float(old.argmax(-1)[active].float().mean()),
                predicted_true_after=float(new.argmax(-1)[active].float().mean()))
        rows.append(dict(span=j, all=summary(mask), gold_false=summary(mask & (gold == 0)),
                         gold_true=summary(mask & (gold == 1)),
                         linearized_weighted_reward_delta=float(linear.mean())))
    linear_sum = sum(r['linearized_weighted_reward_delta'] for r in rows)
    actual = float((new_reward-old_reward).mean())
    return dict(requests=len(targets), spans=rows,
                mean_reward_before=float(old_reward.mean()), mean_reward_after=float(new_reward.mean()),
                reward_delta=actual, linearized_reward_delta=linear_sum,
                nonlinear_remainder=actual-linear_sum)
