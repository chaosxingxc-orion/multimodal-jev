"""Validate padded pooling outputs without inventing categorical actions."""
import torch


def unpack_candidate_logits(logits, candidate_ids):
    """Convert native pooler's ragged padded logits to exact per-Span policies.

    Padding must be negative infinity and never enters a candidate softmax.
    Returns CPU float32 tensors, matching HF candidate normalization.
    """
    if not candidate_ids or any(not isinstance(ids,(list,tuple)) or len(ids)<2 or
            len(set(ids))!=len(ids) or any(type(i) is not int or i<0 for i in ids) for ids in candidate_ids):
        raise ValueError('distinct nonnegative candidate IDs required')
    if not isinstance(logits,torch.Tensor) or logits.ndim!=2 or not logits.is_floating_point():
        raise ValueError('floating two-dimensional pooler logits required')
    widths=[len(ids) for ids in candidate_ids]
    if tuple(logits.shape)!=(len(widths),max(widths)):
        raise ValueError('exact padded pooler shape required')
    output=[]
    for index,width in enumerate(widths):
        valid=logits[index,:width]
        if not torch.isfinite(valid).all():raise ValueError('nonfinite business candidate logit')
        padding=logits[index,width:]
        if padding.numel() and not torch.isneginf(padding).all():
            raise ValueError('only negative-infinity candidate padding allowed')
        output.append(torch.log_softmax(valid.detach().float().cpu(),dim=-1))
    return tuple(output)
