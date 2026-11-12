"""One paired marginal update; caller owns provenance, export and reload gates."""
import torch
from .span_actor_update import admit_actor
from .span_pair_credit import expected_pair_reward,exact_pair_credit
from .span_pair_verl import verl_pair_loss


def update_pair_actor(rollout,actor_forward,optimizer,parameters,targets,pairs,
                      pair_weight,policy_id,request_ids):
    parameters=list(parameters)
    if not parameters or any(not p.requires_grad or p.dtype!=torch.bfloat16 for p in parameters):
        raise ValueError('trainable BF16 execution parameters required')
    behavior=rollout.candidate_logprobs[-1]
    exact_pair_credit(behavior,targets,pairs,pair_weight)
    actor=actor_forward()
    admission=admit_actor(rollout,actor,policy_id,request_ids)
    device=actor[-1].device
    optimizer.zero_grad(set_to_none=True)
    loss,metrics=verl_pair_loss(behavior.to(device),actor[-1],targets.to(device),pairs.to(device),pair_weight)
    if not torch.isfinite(loss):raise ValueError('finite pair loss required')
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=float(loss.detach()),gradient_norm=float(norm),admission=admission,
        metrics={k:float(v) for k,v in metrics.items()},optimizer_updates=1,
        pair_weight=pair_weight,auxiliary_weight=0.,training_requests=len(targets),pairs=len(pairs),
        expected_reward_before=float(expected_pair_reward(behavior,targets,pairs,pair_weight).mean()),
        algorithm='exact-complement-pair-marginal-ppo-v1',sampled_actions_used_for_loss=0,
        scope='Single-rank paired marginal verl loss. Export, independent reload and full runtime gates pending.')
