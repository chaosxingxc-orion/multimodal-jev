"""Scope length grouping to the diagnostic language attention, not its projections."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

import torch
from torch.nn import functional as F

_ACTIVE = ContextVar("single_card_attention_lengths", default=False)


@contextmanager
def by_length(*, contiguous=False):
    token = _ACTIVE.set('contiguous' if contiguous else True)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def active():
    return _ACTIVE.get()


@lru_cache(maxsize=16)
def causal_mask(device):
    return torch.ones((2048, 2048), device=device, dtype=torch.bool).triu(1)


def grouped(q, k, v, lengths, scaling, kernel):
    # Same TND geometry as the equal-length reference. All linear, norm and
    # MLP work continues to use the full batch; restore the original row order.
    if active() == 'contiguous':
        from itertools import groupby
        if list(lengths) != sorted(lengths):
            raise ValueError('contiguous attention requires length-sorted inputs')
        results = []
        start = 0
        for width, members in groupby(lengths):
            count = sum(1 for _ in members)
            stop = start + count
            packed = [t[start:stop, :, :width].transpose(1, 2).reshape(count * width, t.shape[1], t.shape[3]).contiguous()
                      for t in (q, k, v)]
            cu = torch.arange(count + 1, dtype=torch.int32) * width
            out = kernel(*packed, cu, cu, max_seqlen_q=width, max_seqlen_k=width,
                         dropout_p=0., softmax_scale=scaling, causal=True)
            rows = out.reshape(count, width, q.shape[1], q.shape[3])
            results.append(F.pad(rows, (0, 0, 0, 0, 0, q.shape[2] - width)))
            start = stop
        return torch.cat(results)
    by_width = {}
    for index, width in enumerate(lengths):
        by_width.setdefault(width, []).append(index)
    outputs = [None] * len(lengths)
    for width, rows in by_width.items():
        packed = [torch.cat([t[i, :, :width].transpose(0, 1) for i in rows])
                  for t in (q, k, v)]
        cu = torch.arange(len(rows) + 1, dtype=torch.int32) * width
        out = kernel(*packed, cu, cu, max_seqlen_q=width, max_seqlen_k=width,
                     dropout_p=0., softmax_scale=scaling, causal=True)
        for i, part in zip(rows, out.split(width), strict=True):
            outputs[i] = F.pad(part, (0, 0, 0, 0, 0, q.shape[2] - width))
    return torch.stack(outputs)
