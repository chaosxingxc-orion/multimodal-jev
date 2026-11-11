"""Finite parallel-Span policy; an action is the complete tuple of Span values.

All categorical distributions condition on the same supplied audio/questions.
No sampled answer is fed into another Span. Joint log probability is a sum,
not the token-mean surrogate used by the historical text generation actor.
"""
import torch


def sample_spans(candidate_logprobs, samples, generator=None):
    if type(samples) is not int or samples < 2 or not candidate_logprobs:
        raise ValueError('at least two complete samples required')
    batch = candidate_logprobs[0].shape[0]
    actions, selected = [], []
    for logp in candidate_logprobs:
        if logp.ndim != 2 or logp.shape[0] != batch or logp.shape[1] < 2 or not torch.isfinite(logp).all():
            raise ValueError('aligned finite categorical log probabilities required')
        if not torch.allclose(logp.logsumexp(-1),torch.zeros(batch,device=logp.device),atol=1e-6,rtol=0):
            raise ValueError('candidate-normalized distribution required')
        index = torch.multinomial(logp.exp(),samples,replacement=True,generator=generator)
        actions.append(index)
        selected.append(logp.gather(1,index))
    # B x G x S; all spans sampled, including those lacking intermediate gold.
    return torch.stack(actions,-1), torch.stack(selected,-1).sum(-1)


def joint_logprobs(candidate_logprobs, actions):
    if actions.dtype != torch.long or actions.ndim != 3 or actions.shape[-1] != len(candidate_logprobs):
        raise ValueError('integer B x G x S actions required')
    selected = []
    for j, logp in enumerate(candidate_logprobs):
        if logp.ndim != 2 or logp.shape[0] != actions.shape[0]:
            raise ValueError('categorical batch mismatch')
        if ((actions[...,j]<0) | (actions[...,j]>=logp.shape[1])).any():
            raise ValueError('action outside candidate support')
        selected.append(logp.gather(1,actions[...,j]))
    return torch.stack(selected,-1).sum(-1)


def verl_joint_loss(old_joint_logprobs, new_joint_logprobs, rewards, request_ids, clip_ratio=0.2):
    """Use verl GRPO and PPO on one whole-tuple action per response.

    request_ids must identify exact input/policy/sampling configuration, not a
    source recording shared by different question templates. Caller must bind
    rollout artifacts and validate HF/vLLM parity before updates are admitted.
    """
    import numpy as np
    from omegaconf import OmegaConf
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, compute_policy_loss_vanilla
    if (old_joint_logprobs.ndim != 2 or old_joint_logprobs.shape != new_joint_logprobs.shape or
            rewards.shape != old_joint_logprobs.shape or old_joint_logprobs.shape[1]<2 or
            len(request_ids)!=old_joint_logprobs.shape[0] or len(set(request_ids))!=len(request_ids)):
        raise ValueError('complete distinct request groups required')
    if not all(torch.isfinite(x).all() for x in (old_joint_logprobs,new_joint_logprobs,rewards)):
        raise ValueError('finite joint policy and rewards required')
    if old_joint_logprobs.requires_grad or rewards.requires_grad:
        raise ValueError('rollout policy and reward must be detached')
    size = old_joint_logprobs.shape[1]
    mask=torch.ones_like(rewards.reshape(-1,1))
    uid=np.array([u for u in request_ids for _ in range(size)],dtype=object)
    advantages,_=compute_grpo_outcome_advantage(rewards.reshape(-1,1),mask,uid,
                                              norm_adv_by_std_in_grpo=True)
    config=OmegaConf.create({'clip_ratio':clip_ratio,'clip_ratio_low':None,'clip_ratio_high':None,
                            'clip_ratio_c':3.0,'global_batch_info':{}})
    return compute_policy_loss_vanilla(old_joint_logprobs.reshape(-1,1),new_joint_logprobs.reshape(-1,1),
                                      advantages,mask,config=config)
