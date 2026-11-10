"""Explicit, scoped CPU dispatch workaround for the pinned verl actor probe."""
from contextlib import contextmanager


def cpu_logprobs(logits, labels, inplace_backward=True):
    import torch
    if logits.device.type!='cpu' or labels.device.type!='cpu':
        raise ValueError('CPU-only log-prob adapter received accelerator tensors')
    if logits.shape[:-1]!=labels.shape or labels.dtype!=torch.long:
        raise ValueError('aligned integer response labels required')
    return torch.log_softmax(logits,dim=-1).gather(-1,labels.unsqueeze(-1)).squeeze(-1)


@contextmanager
def native_actor_cpu_logprobs():
    """Opt-in replacement of the actor's imported dispatcher, restored on exit.

    Only use in an isolated CPU process with no concurrent actor threads. This
    does not modify the installed package or claim an unmodified worker run.
    """
    from verl.workers.actor import dp_actor
    original=dp_actor.logprobs_from_logits
    dp_actor.logprobs_from_logits=cpu_logprobs
    try:
        yield
    finally:
        dp_actor.logprobs_from_logits=original
