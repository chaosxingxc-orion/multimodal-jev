"""Exact post-update master/Adam comparison for a no-learning-factor replay."""


def compare_states(reference,replay):
    import torch
    for key in ('rank','world_size','updates','names'):
        if reference[key]!=replay[key]:raise ValueError('replay topology/cursor differs: '+key)
    if reference['world_size']!=4 or reference['updates']!=2:
        raise ValueError('fixed four-rank second-update replay required')
    if reference['parameters'].keys()!=replay['parameters'].keys():raise ValueError('parameter coverage differs')
    a,b=reference['optimizer'],replay['optimizer']
    if a['param_groups']!=b['param_groups'] or a['state'].keys()!=b['state'].keys():
        raise ValueError('optimizer groups/state coverage differs')
    comparisons=[]
    for name,p in reference['parameters'].items():comparisons.append(('master:'+name,p,replay['parameters'][name]))
    for k,state in a['state'].items():
        if state.keys()!=b['state'][k].keys():raise ValueError('optimizer fields differ')
        for field,p in state.items():comparisons.append((f'Adam:{k}:{field}',p,b['state'][k][field]))
    for label,p,q in comparisons:
        if (not isinstance(p,torch.Tensor) or not isinstance(q,torch.Tensor) or p.dtype!=torch.float32
                or q.dtype!=p.dtype or p.shape!=q.shape or not torch.isfinite(p).all()
                or not torch.isfinite(q).all() or not torch.equal(p.cpu(),q.cpu())):
            raise ValueError('post-replay tensor differs: '+label)
    return dict(passed=True,rank=reference['rank'],updates=2,exact_master_and_adam=True,
                tensors_compared=len(comparisons),elements=sum(p.numel() for _,p,_ in comparisons))
