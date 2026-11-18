"""Bound only the TND kernel's request count, preserving native service batches.

Candidate, not an admitted runtime. Complete independent causal prefills only.
The wrapper preserves autograd and never pads or combines across boundaries.
"""
import torch

from .span_packed_attention import masked_row_varlen_kernel


def bounded_prefill_kernel(q, k, v, cq, ck, *, max_requests=8, kernel=None, **kwargs):
    if (type(max_requests) is not int or max_requests < 1 or cq.ndim != 1 or ck.ndim != 1
            or cq.dtype not in (torch.int32, torch.int64) or ck.dtype != cq.dtype
            or not torch.equal(cq, ck) or len(cq)<2 or int(cq[0]) != 0
            or int(cq[-1]) != len(q) or len(k)!=len(q) or v.shape != k.shape
            or not bool((cq[1:] > cq[:-1]).all())
            or kwargs.get('causal') is not True or kwargs.get('dropout_p') != 0.):
        raise ValueError('complete aligned causal zero-dropout prefill segments required')
    kernel = masked_row_varlen_kernel if kernel is None else kernel
    ends = cq.tolist(); parts = []
    for start in range(0, len(ends)-1, max_requests):
        stop = min(start+max_requests,len(ends)-1); begin,end = ends[start],ends[stop]
        local = cq[start:stop+1]-begin
        lengths = local[1:]-local[:-1]
        call = dict(kwargs,max_seqlen_q=int(lengths.max()),max_seqlen_k=int(lengths.max()))
        parts.append(kernel(q[begin:end].contiguous(),k[begin:end].contiguous(),
            v[begin:end].contiguous(),local,local,**call))
    return torch.cat(parts,0)
