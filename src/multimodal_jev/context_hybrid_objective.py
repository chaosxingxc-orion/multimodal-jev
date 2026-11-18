"""Compose the pinned sampled verl objective and query-level contrastive terms.

Caller computes advantages from complete groups before splitting DP/microbatches.
No extra backbone calls per sampled action: one B x candidates logit tensor
supports all G binary tuples. Native actor integration is a separate layer.
"""
from .context_binary_policy import binary_logprobs
from .context_candidate_objective import loss as auxiliary_loss
from .span_sampled_hybrid import loss as sampled_loss


def loss(logits, labels, observed, request_weights, real_requests, *,
         stage, global_real_requests, world_size=1,
         ranking_coefficient, probability_coefficient, actions=None,
         old_joint=None, advantages=None, reference_policy=None,
         kl_coefficient=0., clip_ratio=.2, probability_score='binary_nll'):
    if stage not in ('instruction','task'):
        raise ValueError('explicit two-stage context objective required')
    aux,metrics=auxiliary_loss(logits,labels,observed,request_weights,real_requests,
        global_real_requests=global_real_requests,world_size=world_size,
        ranking_coefficient=ranking_coefficient,probability_coefficient=probability_coefficient,
        probability_score=probability_score)
    if stage=='instruction':
        if any(x is not None for x in (actions,old_joint,advantages,reference_policy)) or kl_coefficient!=0:
            raise ValueError('instruction stage has no invented rollout or reference')
        return aux,dict(auxiliary=metrics,stage=stage)
    if any(x is None for x in (actions,old_joint,advantages)):
        raise ValueError('task stage requires actual complete rollout groups')
    policy=binary_logprobs(logits)
    # Disable the old categorical CE contribution: the explicit auxiliary
    # objective above owns ranking and probability supervision exactly once.
    rl,rl_metrics=sampled_loss(policy,actions,old_joint,advantages,labels.long(),observed,
        logits.new_ones(logits.shape),request_weights,real_requests,
        global_real_requests=global_real_requests,world_size=world_size,coefficient=0.,
        reference_policy=reference_policy,kl_coefficient=kl_coefficient,clip_ratio=clip_ratio)
    return rl+aux,dict(auxiliary=metrics,sampled=rl_metrics,stage=stage)
