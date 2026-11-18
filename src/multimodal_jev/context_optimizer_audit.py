"""Validate FP32 original-parameter Adam shards, including unowned empties."""
import torch


def audit_adam_shards(named_parameters,optimizer,step):
    if type(step) is not int or step<1:
        raise ValueError('positive completed update required')
    report={}
    for name,p in named_parameters:
        if p.dtype!=torch.float32 or not torch.isfinite(p).all():
            raise ValueError('finite FP32 master required: '+name)
        state=optimizer.state.get(p,{})
        # FSDP use_orig_params exposes zero-length local views for parameters
        # entirely owned by other ranks. Adam need not allocate moments for them.
        if p.numel()==0 and not state:
            report[name]=dict(numel=0,state='unowned_empty');continue
        if set(state)!={'step','exp_avg','exp_avg_sq'}:
            raise ValueError('complete Adam state required for owned parameter: '+name)
        if state['step'].numel()!=1 or state['step'].dtype!=torch.float32 or state['step'].item()!=step:
            raise ValueError('Adam step differs: '+name)
        for key in ('exp_avg','exp_avg_sq'):
            v=state[key]
            if v.dtype!=torch.float32 or v.shape!=p.shape or not torch.isfinite(v).all():
                raise ValueError('finite aligned FP32 moments required: '+name)
        if (state['exp_avg_sq']<0).any():raise ValueError('negative Adam second moment')
        report[name]=dict(numel=p.numel(),state='initialized',step=step)
    return report
