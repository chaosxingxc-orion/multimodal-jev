"""Complete candidate-set F1 utility for sampled binary Jev actions.

This rewards the decision set, not calibrated marginal probabilities. Missing
labels cannot define set quality: callers must provide a separate supported
task, rather than silently treating missing candidates as negative.
"""
import torch


def set_f1(actions, labels, observed):
    if (actions.ndim != 3 or min(actions.shape) < 1 or actions.dtype != torch.long or
            labels.dtype != torch.bool or observed.dtype != torch.bool or
            labels.shape != (actions.shape[0],actions.shape[2]) or observed.shape != labels.shape or
            any(t.device != actions.device for t in (labels,observed)) or
            ((actions != 0) & (actions != 1)).any()):
        raise ValueError('binary B x samples x candidates actions and aligned Boolean labels required')
    if not observed.all():
        raise ValueError('complete set truth required; missing labels are not negatives')
    selected=actions.bool()
    positive=labels[:,None,:]
    tp=(selected & positive).sum(-1)
    count=selected.sum(-1)+positive.sum(-1)
    reward=torch.where(count>0,2*tp.float()/count.clamp_min(1),torch.ones_like(count,dtype=torch.float32))
    return reward.detach()
