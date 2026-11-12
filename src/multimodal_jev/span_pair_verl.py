"""Experimental marginal clipped pair credit; no existing trainer admission.

Independent per-request ratios, not a joint-pair PPO ratio. Exact-support
quadrature uses rollout_is_weights; backend mismatch is never corrected here.
"""
import math
import torch
from .span_pair_credit import validate,exact_pair_credit


def loss_inputs(behavior,actor,targets,pairs,pair_weight,clip_ratio=.2):
    validate(actor,targets,pairs,pair_weight)
    if (type(clip_ratio) not in (int,float) or not math.isfinite(clip_ratio)
            or not 0<clip_ratio<1):
        raise ValueError('bounded clipping ratio required')
    advantage=exact_pair_credit(behavior,targets,pairs,pair_weight)
    # token mean divides by requests*2 = pairs*4. Multiply quadrature by 4
    # to recover a mean over pairs of sums over requests and candidates.
    return (behavior.reshape(-1,1),actor.reshape(-1,1),advantage.reshape(-1,1),
            behavior.exp().reshape(-1,1)*4)


def reference_pair_loss(behavior,actor,targets,pairs,pair_weight,clip_ratio=.2):
    old,new,adv,weight=loss_inputs(behavior,actor,targets,pairs,pair_weight,clip_ratio)
    ratio=(new-old).clamp(-20,20).exp()
    clipped=torch.maximum(-adv*ratio,-adv*ratio.clamp(1-clip_ratio,1+clip_ratio))
    dual=torch.where(adv<0,torch.minimum(-adv*3.,clipped),clipped)
    return (dual*weight).mean()


def verl_pair_loss(behavior,actor,targets,pairs,pair_weight,clip_ratio=.2):
    from omegaconf import OmegaConf
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
    old,new,adv,weights=loss_inputs(behavior,actor,targets,pairs,pair_weight,clip_ratio)
    config=OmegaConf.create(dict(clip_ratio=clip_ratio,clip_ratio_low=None,clip_ratio_high=None,
                                clip_ratio_c=3.,global_batch_info={}))
    return compute_policy_loss_vanilla(old,new,adv,torch.ones_like(adv),config=config,
                                      rollout_is_weights=weights)
