"""Exact categorical credit diagnostic; not an alternative admitted RL loss.

Compare the sampled score-function estimator of an additive reward against
its exact expectation. Shared-backbone parameter gradients are not equal to
these independent-logit coordinates, and sample-standardized GRPO is biased
relative to this unnormalized expected-reward objective.
"""
import torch
from .span_rule_objectives import validate_rule_targets


def exact_additive_logit_gradient(logprobs, targets, supervised, auxiliary_weight=0.0):
    validate_rule_targets(targets,supervised,len(logprobs),auxiliary_weight)
    weights=supervised.float()
    weights[:,-1]=0
    weights=weights/weights.sum(-1,keepdim=True).clamp_min(1)*auxiliary_weight
    weights[:,-1]=1
    gradients=[]
    for j,logp in enumerate(logprobs):
        if (logp.shape!=(len(targets),2) or not torch.isfinite(logp).all()
                or not torch.allclose(logp.logsumexp(-1),logp.new_zeros(len(targets)),atol=1e-6,rtol=0)):
            raise ValueError('normalized Boolean candidate distributions required')
        probability=logp.exp()
        one_hot=torch.nn.functional.one_hot(targets[:,j].clamp_min(0),2).to(probability)
        correct=(one_hot*probability).sum(-1,keepdim=True)
        gradients.append(weights[:,j:j+1]*correct*(one_hot-probability))
    return torch.stack(gradients,1)


def sampled_logit_gradient(logprobs, actions, rewards, baseline):
    """Per-request ascent gradient; independent baseline must not use sample gold."""
    if actions.ndim!=3 or rewards.shape!=actions.shape[:2] or baseline.shape!=(actions.shape[0],):
        raise ValueError('aligned actions, rewards and request baselines required')
    if actions.dtype!=torch.long or ((actions<0)|(actions>1)).any():raise ValueError('Boolean actions required')
    advantage=rewards-baseline[:,None]
    return torch.stack([((torch.nn.functional.one_hot(actions[:,:,j],2).to(p)-p.exp()[:,None,:])
                          *advantage[:,:,None]).mean(1) for j,p in enumerate(logprobs)],1)
