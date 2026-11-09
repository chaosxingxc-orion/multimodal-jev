"""Fixed reference policy identity and verl's sampled-token KL regularizer."""
import math
from .policy_identity import checkpoint_identity


def reference_identity(config):
    if config is None:
        return None
    if not isinstance(config,dict) or set(config)!={'checkpoint','coefficient','estimator'}:
        raise ValueError('reference requires checkpoint, coefficient and estimator')
    coefficient=config['coefficient']
    if type(coefficient) not in (float,int) or not math.isfinite(coefficient) or coefficient<=0:
        raise ValueError('reference coefficient must be finite and positive; omit reference to disable')
    if config['estimator'] not in {'k2','k3'}:
        raise ValueError('reference estimator must be k2 or k3')
    policy,files=checkpoint_identity(config['checkpoint'])
    return {'policy_id':policy,'checkpoint_files':files,
            'coefficient':coefficient,'estimator':config['estimator']}


def validate_reference_continuation(prior,current):
    if prior!=current:
        raise ValueError('reference policy or regularization changed within trajectory')


def reference_loss(log_probs,reference_log_probs,response_mask,estimator):
    import torch
    from verl.trainer.ppo.core_algos import kl_penalty,agg_loss
    if estimator not in {'k2','k3'}:raise ValueError('unsupported reference estimator')
    if log_probs.shape!=reference_log_probs.shape or log_probs.shape!=response_mask.shape:
        raise ValueError('reference log-prob/mask shape mismatch')
    if reference_log_probs.requires_grad:raise ValueError('reference must be detached')
    mask=response_mask.bool()
    if not mask.any() or not torch.all((response_mask==0)|(response_mask==1)):
        raise ValueError('nonempty binary response mask required')
    if not torch.isfinite(log_probs[mask]).all() or not torch.isfinite(reference_log_probs[mask]).all():
        raise ValueError('nonfinite reference/actor response log-prob')
    # Mask before exponentiation: ignored padding must never introduce NaN.
    actor=torch.where(mask,log_probs,0.)
    reference=torch.where(mask,reference_log_probs,0.)
    penalty=kl_penalty(actor,reference,estimator)
    return agg_loss(penalty,response_mask,loss_agg_mode='token-mean')
