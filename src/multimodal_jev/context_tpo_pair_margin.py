"""Actor-side ranking correction for a pinned Bernoulli projection."""
import math

import torch


def validate_spec(spec):
    margin, weight = spec["logit_margin"], spec["weight"]
    if not math.isfinite(margin) or margin <= 0 or not math.isfinite(weight) or weight < 0:
        raise ValueError("invalid actor pair margin")


@torch.no_grad()
def compile_pairs(reference, targets, gold, approved, margin):
    """Pick the fixed rival and the eligible exposures before actor training."""
    truth = torch.nn.functional.one_hot(gold, reference.shape[-1]).bool()
    wrong_reference = reference.masked_fill(truth, -torch.inf)
    best_wrong, rival = wrong_reference.max(-1)
    strict_error = best_wrong > reference.gather(-1, gold[:, None]).squeeze(-1)
    index = gold[None, :, None].expand(targets.shape[0], -1, 1)
    gold_target = targets.gather(-1, index).squeeze(-1)
    best_wrong_target = targets.masked_fill(truth[None], -torch.inf).amax(-1)
    return rival, approved[None] & strict_error[None] & (gold_target - best_wrong_target < margin)


def loss(logits, targets, gold, rival, enabled, real, spec):
    logits = logits.float()
    columns = torch.stack((gold, rival), dim=-1)
    gold_logit, rival_logit = logits.gather(-1, columns).unbind(-1)
    gap = gold_logit - rival_logit
    eligible = enabled & real
    weight = spec["weight"]
    ranking = torch.where(eligible, (spec["logit_margin"] - gap).relu() * weight, 0.)
    with torch.no_grad():
        active = eligible & (gap < spec["logit_margin"]) & (weight > 0)
        gold_grad, rival_grad = (logits.gather(-1, columns).sigmoid()
                                 - targets.detach().float().gather(-1, columns).sigmoid()).unbind(-1)
        base_signal = (rival_grad - gold_grad) / logits.shape[-1]
        corrected = base_signal + 2 * weight
        counts = {
            "eligible": eligible.sum(),
            "active": active.sum(),
            "turned_positive": (active & (base_signal <= 0) & (corrected > 0)).sum(),
            "strengthened": (active & (base_signal > 0)).sum(),
            "still_nonpositive": (active & (corrected <= 0)).sum(),
        }
    return ranking, counts


def verify_update(row, batch_real, epochs, expected_eligible):
    metrics = row["metrics"]
    base = metrics["jev/reward_projection_kl"]
    pair = metrics["jev/pair_margin"]
    total = metrics["jev/tpo_loss"]
    counts = row["pair_margin"]
    step = row["step"] - 1
    eligible, active, turned, stronger, still = (
        counts[k] for k in ("eligible", "active", "turned_positive", "strengthened", "still_nonpositive")
    )
    if (row["real"] != batch_real[step] or row["epoch"] != epochs[step]
            or eligible != expected_eligible[step] or not 0 <= active <= eligible <= row["real"]
            or not all(type(n) is int and n >= 0 for n in (eligible, active, turned, stronger, still))
            or active != turned + stronger + still
            or not all(math.isfinite(v) for v in (base, pair, total))
            or base <= 0 or pair < 0 or (pair > 0) != (active > 0)
            or not math.isclose(total, base + pair, rel_tol=1e-6, abs_tol=1e-9)):
        raise ValueError("invalid actor pair margin update")
