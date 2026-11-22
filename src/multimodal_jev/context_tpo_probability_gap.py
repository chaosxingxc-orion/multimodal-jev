"""A probability-space correction against the current actor's strongest wrong label."""
import math
import torch


def probability_gap(logits, targets, gold, reference_rival, real, spec):
    p = logits.float().sigmoid()
    wrong = p.scatter(1, gold[:, None], -1.)
    rival = wrong.argmax(-1)
    columns = torch.stack((gold, rival), dim=-1)
    pos, neg = p.gather(1, columns).unbind(-1)
    gap = pos - neg
    penalty = (spec["margin"] - gap).relu() * spec["weight"]
    with torch.no_grad():
        q_pos, q_neg = targets.float().sigmoid().gather(1, columns).unbind(-1)
        base = ((neg-q_neg) - (pos-q_pos)) / p.shape[-1]
        active = gap < spec["margin"]
        extra = (pos*(1-pos) + neg*(1-neg)) * spec["weight"] * active
        counts = dict(active=(active & real).sum(), real=real.sum(),
                      rival_switch=((rival != reference_rival) & real).sum(),
                      projected_correct=((targets.argmax(-1)==gold) & real).sum())
        diagnostics = dict(base_gap_gradient=base, abs_base_gap_gradient=base.abs(),
                           margin_gap_gradient=extra)
    return penalty, counts, diagnostics


def verify_gap_update(row, step, batch_real):
    metrics = row["metrics"]
    base, penalty, total = (metrics[k] for k in ("jev/reward_projection_kl","jev/probability_gap","jev/tpo_loss"))
    counts = row["gap_counts"]
    if (row["step"] != step or row["real"] != batch_real[step-1] or counts["real"] != row["real"]
            or any(not isinstance(n,int) or not 0 <= n <= row["real"] for n in counts.values())
            or not all(math.isfinite(v) for v in metrics.values()) or not base > 0 or penalty < 0
            or (counts["active"] > 0) != (penalty > 0) or metrics["actor/grad_norm"] <= 0
            or not math.isclose(total,base+penalty,rel_tol=1e-6,abs_tol=1e-9)):
        raise ValueError("invalid probability-gap update")
