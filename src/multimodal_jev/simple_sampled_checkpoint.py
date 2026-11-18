"""Independent reconstruction of the fixed DP4 sampled-RL checkpoint.

No acceptance of training efficacy or backend probabilities is implied.
"""


def reconstruct(states, exported, *, algorithm, adapter_id, reference_policy_id, lr,
                expected_names, completed_updates=1, expected_stage='sampled_rl_engineering'):
    import torch
    from .span_adapter_export import adapter_key
    if (algorithm not in ('grpo_v1', 'rloo_v1') or len(states) != 4
            or expected_stage not in ('sampled_rl_engineering','continuous_rl_engineering','simple_stage2_rl')
            or type(completed_updates) is not int
            or (completed_updates not in (1,2) if expected_stage=='sampled_rl_engineering'
                else not 1<=completed_updates<=634)):
        raise ValueError('registered sampled algorithm and four shards required')
    if len(expected_names) != 112 or len(set(expected_names)) != 112:
        raise ValueError('independently bound 112-parameter layout required')
    expected_group = dict(lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0.,amsgrad=False,
        maximize=False,foreach=False,capturable=False,differentiable=False,fused=False,
        decoupled_weight_decay=True,params=list(range(112)))
    for rank,state in enumerate(states):
        expected=dict(rank=rank,world_size=4,updates=completed_updates,stage=expected_stage,
                      algorithm=algorithm,adapter_id=adapter_id,reference_policy_id=reference_policy_id)
        if any(state.get(k)!=v for k,v in expected.items()):
            raise ValueError('sampled checkpoint lineage or cursor differs')
        if state['names']!=expected_names or set(state['parameters'])!=set(expected_names):
            raise ValueError('sampled checkpoint parameter layout differs')
        optimizer=state['optimizer']
        if optimizer['param_groups']!=[expected_group] or set(optimizer['state'])!=set(range(112)):
            raise ValueError('complete fixed Adam groups and state required')
        for i,name in enumerate(expected_names):
            master=state['parameters'][name];adam=optimizer['state'][i]
            if master.dtype!=torch.float32 or master.ndim!=1 or not torch.isfinite(master).all():
                raise ValueError('finite flat FP32 master required')
            if set(adam)!={'step','exp_avg','exp_avg_sq'}:
                raise ValueError('complete Adam fields required')
            step=adam['step']
            if step.dtype!=torch.float32 or step.numel()!=1 or step.item()!=completed_updates:
                raise ValueError('exact declared Adam cursor required')
            for field in ('exp_avg','exp_avg_sq'):
                value=adam[field]
                if value.dtype!=torch.float32 or value.shape!=master.shape or not torch.isfinite(value).all():
                    raise ValueError('finite shape-matched FP32 moments required')
            if (adam['exp_avg_sq']<0).any():raise ValueError('negative second moment')
    reconstructed={}
    for name in expected_names:
        key=adapter_key(name,native=True)
        if key in reconstructed or key not in exported:raise ValueError('exact export layout required')
        value=exported[key];master=torch.cat([s['parameters'][name] for s in states])
        if master.numel()!=value.numel() or value.dtype!=torch.bfloat16 or not torch.isfinite(value).all():
            raise ValueError('BF16 export shape/dtype differs')
        expected=master.reshape(value.shape).to(torch.bfloat16)
        if not torch.equal(expected,value):raise ValueError('master reconstruction differs from export')
        reconstructed[key]=expected
    if reconstructed.keys()!=exported.keys():raise ValueError('unexpected export tensor')
    return reconstructed
