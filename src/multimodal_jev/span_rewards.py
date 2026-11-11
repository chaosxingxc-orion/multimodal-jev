"""Audited finite decision rewards; no invented textual process scores."""
import math
import torch


def decision_rewards(actions, targets, supervised, final_index, auxiliary_weight=0.0):
    """B x G x S candidate-index actions; final gold mandatory for this arm.

    Intermediate reward is a masked mean, never grows with question count.
    Unknown is an ordinary legal candidate supplied by the caller. Missing gold
    is represented only by supervised=False. This is not evidence of causal CoT.
    """
    if (actions.ndim!=3 or actions.dtype!=torch.long or targets.dtype!=torch.long or
            targets.shape!=(actions.shape[0],actions.shape[2]) or supervised.shape!=targets.shape or
            supervised.dtype!=torch.bool or type(final_index)!=int or
            not 0<=final_index<actions.shape[2]):
        raise ValueError('aligned typed actions, targets and masks required')
    if (not math.isfinite(auxiliary_weight) or not 0<=auxiliary_weight<1 or
            (actions<0).any() or (targets[supervised]<0).any()):
        raise ValueError('valid candidate indices and bounded auxiliary weight required')
    if not supervised[:,final_index].all():
        raise ValueError('Final-reward arm requires reliable Final supervision')
    correct=(actions==targets[:,None,:]).float()
    final=correct[:,:,final_index]
    aux_mask=supervised.clone();aux_mask[:,final_index]=False
    count=aux_mask.sum(-1)
    auxiliary=(correct*aux_mask[:,None,:]).sum(-1)/count.clamp_min(1)[:,None]
    auxiliary=torch.where(count[:,None]>0,auxiliary,torch.zeros_like(auxiliary))
    return dict(total=final+auxiliary_weight*auxiliary,final=final,auxiliary=auxiliary,
                auxiliary_supervised_counts=count)
