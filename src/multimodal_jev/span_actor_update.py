"""Single-rank Span actor update with mandatory behavior-policy admission.

Consumes native vLLM candidate probabilities, never generated token logprobs.
The caller owns authenticated input/adapter loading and the rollout barrier.
This is an optimizer adapter, not a replacement for the verl trainer lifecycle.
"""
from dataclasses import dataclass
import torch

from .span_parity import compare_span_policies
from .span_policy import sample_spans, joint_logprobs, verl_joint_loss


@dataclass(frozen=True)
class SpanRollout:
    policy_id: str
    request_ids: tuple[str, ...]
    candidate_logprobs: tuple[torch.Tensor, ...]
    actions: torch.Tensor
    old_joint_logprobs: torch.Tensor


def record_rollout(policy_id, request_ids, candidate_logprobs, samples, generator=None):
    if not isinstance(policy_id,str) or not policy_id:
        raise ValueError('immutable behavior policy identity required')
    request_ids=tuple(request_ids)
    if (not request_ids or any(not isinstance(s,str) or not s for s in request_ids)
            or len(set(request_ids))!=len(request_ids)):
        raise ValueError('distinct exact input request identities required')
    saved=tuple(p.detach().cpu().float().clone() for p in candidate_logprobs)
    if not saved or any(p.ndim!=2 or p.shape[0]!=len(request_ids) for p in saved):
        raise ValueError('request-aligned candidate distributions required')
    actions,old=sample_spans(saved,samples,generator)
    return SpanRollout(policy_id,request_ids,saved,actions,old)


def admit_actor(rollout, actor_logprobs, policy_id, request_ids):
    """Fail before backward if any input/version/distribution differs."""
    if policy_id!=rollout.policy_id or tuple(request_ids)!=rollout.request_ids:
        raise ValueError('stale policy or different exact input request')
    # Recompute from the actual saved behavior distribution only. Never replace
    # behavior probabilities with actor recomputation to manufacture ratio=1.
    expected=joint_logprobs(rollout.candidate_logprobs,rollout.actions)
    if (rollout.old_joint_logprobs.requires_grad or
            not torch.equal(expected,rollout.old_joint_logprobs)):
        raise ValueError('behavior joint logprobs were changed')
    if len(actor_logprobs)!=len(rollout.candidate_logprobs):
        raise ValueError('Span support changed')
    for actor,behavior in zip(actor_logprobs,rollout.candidate_logprobs):
        if actor.shape!=behavior.shape:
            raise ValueError('candidate dimensions changed')
    audits=[]
    for i in range(len(rollout.request_ids)):
        audit=compare_span_policies([p[i].detach().cpu().tolist() for p in actor_logprobs],
                                    [p[i].tolist() for p in rollout.candidate_logprobs])
        audits.append(audit)
    if not all(a['passed'] for a in audits):
        raise ValueError('actor/behavior complete-action parity gate failed')
    return audits


def update_span_actor(rollout, actor_forward, optimizer, parameters, rewards,
                      policy_id, request_ids, clip_ratio=0.2):
    """One full request-group verl GRPO/PPO step, no microbatch reweighting.

Only call once per rollout version. Export a new immutable BF16 adapter after
this returns, reload it in vLLM, and validate the next version before sampling.
Rewards must come from the audited task scorer outside model input construction.
"""
    if (rewards.requires_grad or rewards.shape!=rollout.old_joint_logprobs.shape
            or not torch.isfinite(rewards).all()):
        raise ValueError('finite detached complete-group rewards required')
    parameters=list(parameters)
    if not parameters or any(not p.requires_grad or p.dtype!=torch.bfloat16 for p in parameters):
        raise ValueError('trainable BF16 adapter execution parameters required')
    actor=actor_forward()
    audits=admit_actor(rollout,actor,policy_id,request_ids)
    device=actor[0].device
    new_joint=joint_logprobs(actor,rollout.actions.to(device))
    old_joint=rollout.old_joint_logprobs.to(device)
    optimizer.zero_grad(set_to_none=True)
    loss,metrics=verl_joint_loss(old_joint,new_joint,rewards.to(device),rollout.request_ids,clip_ratio)
    if not torch.isfinite(loss):
        raise ValueError('nonfinite verl Span loss')
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=float(loss.detach()),gradient_norm=float(norm),
                admission=audits,optimizer_updates=1,behavior_policy_id=policy_id,
                next_action='export new BF16 policy identity; reload and verify worker before next rollout',
                scope='Single-rank native verl joint loss/optimizer adapter; caller owns lifecycle and provenance.')
