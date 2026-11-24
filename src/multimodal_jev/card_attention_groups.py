"""Scope length grouping to the diagnostic language attention, not its projections."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

import torch
from torch.nn import functional as F

_ACTIVE = ContextVar("single_card_attention_lengths", default=False)


@contextmanager
def by_length():
    token = _ACTIVE.set(True)
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
