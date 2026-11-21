"""Optional supervised Stage1 terms for paired complete native decision panels.

Paired rows must have been co-located by the sampler. Native logits remain
independent sigmoid readouts; margins here are training-only.
"""
import torch
from torch.nn import functional as F


def _index(index, rows, upper):
    if (index.shape!=(rows,) or index.dtype!=torch.long or
            ((index<0)|(index>=upper)).any()):
        raise ValueError('one valid semantic index per recording required')


def cue_pair_loss(up, down, positive, contrast, *, temperature, probability_margin):
    """Return per-pair values. Both utterances must independently beat their foil."""
    if (up.ndim!=2 or up.shape!=down.shape or up.shape[0]<1 or up.shape[1]<2
            or not torch.isfinite(up).all() or not torch.isfinite(down).all()
            or not (0<temperature<float('inf') and 0<=probability_margin<1)):
        raise ValueError('finite paired panels and valid contrast parameters required')
    n,k=up.shape
    _index(positive,n,k);_index(contrast,n,k)
    if (positive==contrast).any():raise ValueError('foil cannot equal target')
    i=torch.arange(n,device=up.device)
    # Inputs have already been mapped back to the same semantic order.
    pos=torch.stack((up[i,positive],down[i,contrast])).float()
    neg=torch.stack((up[i,contrast],down[i,positive])).float()
    terms=F.softplus((neg-pos)/temperature)+F.relu(probability_margin-(pos.sigmoid()-neg.sigmoid()))
    return terms.mean(0)


def remap_loss(primary, permuted, positions, *, labels):
    """Per-audio semantic invariance plus supervised native ranking on the second view.

    positions[i,k] gives the location of semantic candidate k in permuted[i].
    A constant predictor must not win by having perfectly consistent outputs.
    """
    if (primary.ndim!=2 or primary.shape!=permuted.shape or positions.shape!=primary.shape
            or positions.dtype!=torch.long or not torch.isfinite(primary).all() or not torch.isfinite(permuted).all()):
        raise ValueError('matching finite paired views and explicit remapping required')
    n,k=primary.shape
    _index(labels,n,k)
    if k<2 or not torch.equal(positions.sort(-1).values,torch.arange(k,device=positions.device).expand(n,k)):
        raise ValueError('bijective mapping required')
    a,b=primary.float(),permuted.float().gather(1,positions)
    return F.smooth_l1_loss(a,b,reduction='none').mean(-1), F.cross_entropy(b,labels,reduction='none')
