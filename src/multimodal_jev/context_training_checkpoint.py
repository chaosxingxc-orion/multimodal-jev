"""Fixed-topology FP32 master/Adam resume for the context training loop.

File hashes and all-rank commit are authenticated by caller. Validate every
local tensor and experiment identity before changing a live model or optimizer.
"""
import copy
import torch

from .context_optimizer_audit import audit_adam_shards


def capture(named_parameters,optimizer,*,rank,world_size,completed_updates,contract):
    named=list(named_parameters)
    audit_adam_shards(named,optimizer,completed_updates)
    return dict(schema='context-training-shard-v1',rank=rank,world_size=world_size,
        completed_updates=completed_updates,contract=copy.deepcopy(contract),
        names=[n for n,p in named],parameters={n:p.detach().cpu().clone() for n,p in named},
        optimizer=copy.deepcopy(optimizer.state_dict()))


def validate(named_parameters,optimizer,state,*,rank,world_size,completed_updates,contract):
    named=list(named_parameters);names=[n for n,p in named]
    if (state.get('schema')!='context-training-shard-v1' or state.get('rank')!=rank or
            state.get('world_size')!=world_size or state.get('completed_updates')!=completed_updates or
            type(completed_updates) is not int or completed_updates<1 or state.get('contract')!=contract or
            state.get('names')!=names or set(state.get('parameters',{}))!=set(names) or len(set(names))!=len(names)):
        raise ValueError('checkpoint topology, objective, cursor or names differ')
    live=optimizer.state_dict();saved=state['optimizer']
    if optimizer.state or saved['param_groups']!=live['param_groups'] or len(live['param_groups'])!=1:
        raise ValueError('fresh optimizer with exact hyperparameters required')
    if [id(p) for p in optimizer.param_groups[0]['params']]!=[id(p) for n,p in named]:
        raise ValueError('optimizer parameter ordering differs')
    keys=live['param_groups'][0]['params']
    if not set(saved['state'])<=set(keys):raise ValueError('unknown optimizer state parameter')
    for (name,p),key in zip(named,keys,strict=True):
        value=state['parameters'][name]
        if (p.dtype!=torch.float32 or not isinstance(value,torch.Tensor) or value.dtype!=torch.float32 or
                value.shape!=p.shape or not torch.isfinite(value).all()):
            raise ValueError('finite exact-shape FP32 master required: '+name)
        moment=saved['state'].get(key,{})
        if p.numel()==0 and not moment:continue
        if set(moment)!={'step','exp_avg','exp_avg_sq'}:
            raise ValueError('owned parameter missing complete Adam state: '+name)
        step=moment['step']
        if (not isinstance(step,torch.Tensor) or step.numel()!=1 or step.dtype!=torch.float32 or
                step.item()!=completed_updates):raise ValueError('Adam update count differs')
        for field in ('exp_avg','exp_avg_sq'):
            t=moment[field]
            if (not isinstance(t,torch.Tensor) or t.shape!=p.shape or t.dtype!=torch.float32 or
                    not torch.isfinite(t).all()):raise ValueError('Adam moment differs')
        if (moment['exp_avg_sq']<0).any():raise ValueError('negative second moment')
    return named


def restore(named_parameters,optimizer,state,**identity):
    named=validate(named_parameters,optimizer,state,**identity)
    with torch.no_grad():
        for name,p in named:p.copy_(state['parameters'][name])
    optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
    actual=optimizer.state_dict()
    for key,fields in state['optimizer']['state'].items():
        for field,value in fields.items():
            if not torch.equal(value.cpu(),actual['state'][key][field].cpu()):
                raise ValueError('restored optimizer differs')
    if any(not torch.equal(p.detach().cpu(),state['parameters'][name]) for name,p in named):
        raise ValueError('restored master differs')
    audit_adam_shards(named,optimizer,identity['completed_updates'])
    return dict(exact_masters=True,exact_adam=True,completed_updates=identity['completed_updates'])
