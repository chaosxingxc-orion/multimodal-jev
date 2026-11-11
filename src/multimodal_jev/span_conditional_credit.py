"""Exact conditional credit for additive, independently sampled Boolean spans.

This is a new marginal PPO surrogate, not algebraically equivalent to joint
PPO clipping away from the behavior policy. No sampled intermediate answer is
fed to Final. Reliable labels are consumed only by the reward scorer.
"""
import math
import torch
from .span_rule_objectives import validate_rule_targets


def conditional_credit(behavior, targets, supervised, auxiliary_weight=0.0):
    """Return detached per-candidate centered reward and exact behavior weights.

    E[R | a_j] - E[R] = w_j (1[a_j=gold_j] - p_j(gold_j))
    for this factorized policy and additive reward. Missing facts receive zero
    weight; Final gold is mandatory. No group sample-standardization is used.
    """
    validate_rule_targets(targets,supervised,len(behavior),auxiliary_weight)
    weights=supervised.float().clone();weights[:,-1]=0
    weights=weights/weights.sum(-1,keepdim=True).clamp_min(1)*auxiliary_weight
    weights[:,-1]=1
    advantages=[];probabilities=[]
    for j,old in enumerate(behavior):
        if (old.requires_grad or old.shape!=(len(targets),2) or not torch.isfinite(old).all()
                or not torch.allclose(old.logsumexp(-1),old.new_zeros(len(targets)),atol=1e-6,rtol=0)):
            raise ValueError('detached normalized Boolean behavior policy required')
        probability=old.exp()
        gold=torch.nn.functional.one_hot(targets[:,j].masked_fill(~supervised[:,j],0),2).to(old)
        mean=(probability*gold).sum(-1,keepdim=True)
        advantages.append((gold-mean)*weights[:,j:j+1])
        probabilities.append(probability)
    return torch.stack(advantages,1).detach(),torch.stack(probabilities,1).detach()


def loss_inputs(behavior,actor,targets,supervised,auxiliary_weight=0.0,clip_ratio=.2):
    if (type(clip_ratio) not in (int,float) or not math.isfinite(clip_ratio) or not 0<clip_ratio<1
            or len(actor)!=len(behavior)):
        raise ValueError('aligned policies and bounded clipping ratio required')
    advantages,probabilities=conditional_credit(behavior,targets,supervised,auxiliary_weight)
    for old,new in zip(behavior,actor,strict=True):
        if (new.shape!=old.shape or not torch.isfinite(new).all()
                or not torch.allclose(new.logsumexp(-1),new.new_zeros(len(targets)),atol=1e-6,rtol=0)):
            raise ValueError('aligned normalized actor probabilities required')
    # verl's token-mean divides by B*S*2. Scale exact old-policy weights by
    # S*2 to obtain mean over requests of SUM over spans/candidates. This keeps
    # the Final scale invariant when unscored intermediate questions are added.
    return (torch.stack(behavior,1).reshape(-1,1),torch.stack(actor,1).reshape(-1,1),
            advantages.reshape(-1,1),probabilities.reshape(-1,1)*(2*len(behavior)))


def reference_conditional_loss(behavior,actor,targets,supervised,auxiliary_weight=0.0,clip_ratio=.2):
    """Manual oracle matching pinned verl vanilla dual-clipped PPO semantics."""
    old,new,adv,weight=loss_inputs(behavior,actor,targets,supervised,auxiliary_weight,clip_ratio)
    ratio=(new-old).clamp(-20,20).exp()
    clipped=torch.maximum(-adv*ratio,-adv*ratio.clamp(1-clip_ratio,1+clip_ratio))
    dual=torch.where(adv<0,torch.minimum(-adv*3.,clipped),clipped)
    return (dual*weight).mean()


def verl_conditional_loss(behavior,actor,targets,supervised,auxiliary_weight=0.0,clip_ratio=.2):
    """Exact-support marginal surrogate through real verl PPO implementation.

    Caller must initialize verl before constructing the HF actor, preserve the
    accepted Qwen3 equations, and authenticate per-request full policy parity.
    rollout_is_weights is used solely for finite-support quadrature here, not
    for correcting actor/rollout drift. Backend drift is always a failed gate.
    """
    from omegaconf import OmegaConf
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
    old,new,adv,weights=loss_inputs(behavior,actor,targets,supervised,auxiliary_weight,clip_ratio)
    config=OmegaConf.create(dict(clip_ratio=clip_ratio,clip_ratio_low=None,clip_ratio_high=None,
                                clip_ratio_c=3.,global_batch_info={}))
    return compute_policy_loss_vanilla(old,new,adv,torch.ones_like(adv),config=config,
                                      rollout_is_weights=weights)
