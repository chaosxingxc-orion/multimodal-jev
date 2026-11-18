"""Sampled whole-action verl PPO surrogate plus once-per-request auxiliary CE.

Caller authenticates behavior, computes full-group advantages before partition,
and validates global real counts collectively. This is not a trainer or rollout.
"""
import inspect
import math
import torch
from .baseline import file_hash
from .span_group_advantage import CORE_SHA256
from .span_policy import joint_logprobs
from .span_finite_credit import validate_policy
from .span_hybrid_objective import supervised_numerator


def loss(policy, actions, old_joint, advantages, targets, supervised, span_weights,
         request_weights, real_requests, *, global_real_requests, coefficient,
         world_size=1, clip_ratio=.2, reference_policy=None, kl_coefficient=0.):
    batch=targets.shape[0]
    validate_policy(policy,batch)
    device=policy[0].device
    if (type(global_real_requests) is not int or global_real_requests<1
            or type(world_size) is not int or world_size<1
            or real_requests.dtype!=torch.bool or real_requests.shape!=(batch,)
            or real_requests.device!=device or int(real_requests.sum())>global_real_requests
            or request_weights.shape!=(batch,) or (request_weights[~real_requests]!=0).any()
            or type(coefficient) not in (int,float) or not math.isfinite(coefficient) or coefficient<0
            or type(clip_ratio) not in (int,float) or not 0<clip_ratio<1):
        raise ValueError('explicit real-count reduction, zero padding and bounded coefficients required')
    if (actions.ndim!=3 or actions.shape[0]!=batch or actions.shape[1]<2
            or old_joint.shape!=actions.shape[:2] or advantages.shape!=old_joint.shape
            or any(t.device!=device for t in (actions,old_joint,advantages))
            or any(t.requires_grad or not t.is_floating_point() or not torch.isfinite(t).all()
                   for t in (old_joint,advantages))):
        raise ValueError('detached complete-group behavior and advantages required')
    if (old_joint>1e-6).any():
        raise ValueError('joint log probabilities cannot be positive')
    # Validates labels and weights even on zero-weight padding, masks missing gold.
    ce_numerator,_=supervised_numerator(policy,targets,supervised,span_weights,request_weights)
    kl_numerator=reference_kl_numerator(policy,reference_policy,request_weights,kl_coefficient)
    new_joint=joint_logprobs(policy,actions)
    from omegaconf import OmegaConf
    from verl.trainer.ppo import core_algos
    if file_hash(inspect.getfile(core_algos))!=CORE_SHA256:
        raise ValueError('pinned native surrogate changed')
    config=OmegaConf.create(dict(clip_ratio=clip_ratio,clip_ratio_low=None,clip_ratio_high=None,
                                clip_ratio_c=3.,global_batch_info={}))
    # Native mean is over B*G whole actions. The outer factor makes this
    # sum_x weight_x * mean_G(loss_x) / Nreal, including zero-signal requests.
    weights=request_weights[:,None].expand_as(old_joint).reshape(-1,1)
    raw,metrics=core_algos.compute_policy_loss_vanilla(
        old_joint.reshape(-1,1),new_joint.reshape(-1,1),advantages.reshape(-1,1),
        torch.ones_like(old_joint).reshape(-1,1),config=config,rollout_is_weights=weights)
    scale=world_size/global_real_requests
    rl=raw*batch*scale
    ce=ce_numerator*scale
    kl=kl_numerator*scale
    return rl+coefficient*ce+kl_coefficient*kl, dict(rl=float(rl.detach()),ce=float(ce.detach()),
        kl=float(kl.detach()),kl_coefficient=kl_coefficient,
        coefficient=coefficient,global_real_requests=global_real_requests,
        local_real_requests=int(real_requests.sum()),group_size=actions.shape[1],native_metrics=metrics)


def reference_kl_numerator(policy, reference_policy, request_weights, coefficient):
    """Sum exact categorical KL once per request, retaining policy gradients.

    Caller binds the detached reference to the fixed stage1 checkpoint; never
    use the latest behavior snapshot as a substitute for that reference.
    """
    if type(coefficient) not in (int,float) or not math.isfinite(coefficient) or coefficient<0:
        raise ValueError('finite nonnegative reference KL coefficient required')
    if reference_policy is None:
        if coefficient!=0:
            raise ValueError('positive KL requires the fixed reference distribution')
        return policy[0].sum()*0
    validate_policy(reference_policy,policy[0].shape[0],detached=True)
    if len(reference_policy)!=len(policy) or any(a.shape!=b.shape or a.device!=b.device
                                                for a,b in zip(policy,reference_policy,strict=True)):
        raise ValueError('reference candidate support differs')
    kl=sum((p.exp()*(p-q)).sum(-1) for p,q in zip(policy,reference_policy,strict=True))
    return (kl*request_weights).sum()
