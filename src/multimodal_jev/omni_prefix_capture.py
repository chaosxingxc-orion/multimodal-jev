"""Stop a diagnostic forward exactly before the first text decoder layer."""
from contextlib import contextmanager


class PrefixCaptured(Exception):
    """Expected local diagnostic stop; never a successful actor probability."""


@contextmanager
def capture_prefix(layer, token_mask, storage):
    import torch
    if layer.training or torch.is_grad_enabled() or storage:
        raise ValueError('eval inference with empty capture storage required')
    if (token_mask.ndim != 2 or not torch.all((token_mask==0)|(token_mask==1))
            or not torch.all(token_mask.sum(-1)>0)):
        raise ValueError('nonempty binary token masks required')
    def clone(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
        if isinstance(value,tuple):return tuple(clone(v) for v in value)
        if value is None:return None
        raise ValueError('unsupported first-layer input type')
    def before(module,args,kwargs):
        hidden = args[0] if args else kwargs['hidden_states']
        if (hidden.ndim!=3 or hidden.shape[:2]!=token_mask.shape
                or not torch.isfinite(hidden).all()):
            raise ValueError('first-layer hidden states differ from token mask')
        storage['hidden_states']=clone(hidden)
        storage['token_mask']=clone(token_mask)
        for name in ('position_ids','position_embeddings','attention_mask'):
            if name in kwargs:storage[name]=clone(kwargs[name])
        raise PrefixCaptured('captured before first text decoder layer')
    handle=layer.register_forward_pre_hook(before,with_kwargs=True)
    try:yield
    finally:handle.remove()
