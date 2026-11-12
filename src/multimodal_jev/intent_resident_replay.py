"""Exact resident/reference state checks; no checkpoint mutation or tolerance."""


def compare_state(reference,actual,*,rank,updates):
    import torch
    if type(rank)!=int or rank not in range(4) or type(updates)!=int or updates not in (1,2):
        raise ValueError('fixed DP4 two-update replay required')
    for state in (reference,actual):
        if (state['rank']!=rank or state['world_size']!=4 or state['updates']!=updates
                or not state['names'] or len(set(state['names']))!=len(state['names'])
                or set(state['parameters'])!=set(state['names'])):
            raise ValueError('complete fixed rank/cursor/layout required')
    if reference['names']!=actual['names']:raise ValueError('parameter ordering differs')
    if reference['optimizer']['param_groups']!=actual['optimizer']['param_groups']:raise ValueError('optimizer groups differ')
    pairs=[(p,actual['parameters'][n]) for n,p in reference['parameters'].items()]
    a,b=reference['optimizer']['state'],actual['optimizer']['state']
    keys=[k for g in reference['optimizer']['param_groups'] for k in g['params']]
    if set(a)!=set(b) or set(a)!=set(keys) or len(keys)!=len(set(keys)):raise ValueError('complete optimizer state required')
    for key,state in a.items():
        if set(state)!={'step','exp_avg','exp_avg_sq'} or set(b[key])!=set(state):raise ValueError('complete Adam fields required')
        if state['step'].numel()!=1 or state['step'].item()!=updates:raise ValueError('Adam cursor differs')
        pairs.extend((p,b[key][field]) for field,p in state.items())
    for p,q in pairs:
        if (not isinstance(p,torch.Tensor) or not isinstance(q,torch.Tensor) or p.dtype!=torch.float32
                or q.dtype!=p.dtype or p.shape!=q.shape or not torch.isfinite(p).all()
                or not torch.isfinite(q).all() or not torch.equal(p.cpu(),q.cpu())):
            raise ValueError('resident master/Adam differs from accepted reference')
    return dict(passed=True,rank=rank,updates=updates,tensors_compared=len(pairs),exact=True)
