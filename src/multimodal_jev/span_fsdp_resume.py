"""Strict fixed-topology restore of native FSDP FP32 LoRA/Adam shards.

The caller authenticates files, base, adapter, runtime and data provenance.
This module validates the live shard layout before changing any parameter.
It deliberately does not migrate single-card optimizers or world sizes.
"""


def validate_cursor(previous, current, world_size):
    for key in ('schema', 'seed', 'schedule_sha256', 'epoch_batches',
                'manifest_sha256', 'question_order'):
        if key not in previous or previous[key] != current.get(key):
            raise ValueError('FSDP resume data schedule changed: ' + key)
    # Each rank consumes one pair, so its next pair advances by world_size.
    if current['batch_index'] != previous['batch_index'] + world_size:
        raise ValueError('FSDP resume must advance the global balanced batch')
    if not 0 <= current['batch_index'] < current['epoch_batches']:
        raise ValueError('FSDP resume exhausted the balanced epoch')


def restore_local_shard(model, optimizer, state, *, rank, world_size, updates):
    import torch

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    names = [n for n, _ in named]
    if (state.get('world_size') != world_size or state.get('rank') != rank
            or state.get('updates') != updates or updates < 1
            or state.get('names') != names
            or set(state.get('parameters', {})) != set(names)):
        raise ValueError('FSDP checkpoint topology, cursor or parameter names differ')
    if optimizer.state:
        raise ValueError('restore requires a fresh optimizer')
    live_groups = optimizer.state_dict()['param_groups']
    saved = state['optimizer']
    if saved['param_groups'] != live_groups or len(live_groups) != 1:
        raise ValueError('FSDP optimizer hyperparameters or groups differ')
    live_parameters = optimizer.param_groups[0]['params']
    if [id(p) for p in live_parameters] != [id(p) for _, p in named]:
        raise ValueError('FSDP optimizer parameter order differs')
    keys = live_groups[0]['params']
    if set(saved['state']) != set(keys):
        raise ValueError('FSDP Adam state coverage differs')
    for (name, parameter), key in zip(named, keys, strict=True):
        master = state['parameters'][name]
        if (parameter.dtype != torch.float32 or master.dtype != torch.float32
                or parameter.shape != master.shape or not torch.isfinite(master).all()):
            raise ValueError('finite FP32 master with exact local shape required')
        # The model was loaded from the accepted BF16 adapter, then sharded.
        # Masters may contain sub-BF16 information that must survive resume.
        if not torch.equal(parameter.detach().cpu().to(torch.bfloat16),
                           master.cpu().to(torch.bfloat16)):
            raise ValueError('FP32 master does not represent loaded BF16 adapter')
        adam = saved['state'][key]
        if set(adam) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('unsupported or incomplete Adam state')
        step = adam['step']
        if (not isinstance(step, torch.Tensor) or step.numel() != 1
                or step.dtype != torch.float32 or step.item() != updates):
            raise ValueError('Adam step differs from completed updates')
        for field in ('exp_avg', 'exp_avg_sq'):
            value = adam[field]
            if (value.shape != parameter.shape or value.dtype != torch.float32
                    or not torch.isfinite(value).all()):
                raise ValueError('finite FP32 Adam moments with exact shape required')
        if (adam['exp_avg_sq'] < 0).any():
            raise ValueError('negative Adam second moment')
    # Validate the entire checkpoint before the first write.
    with torch.no_grad():
        for name, parameter in named:
            parameter.copy_(state['parameters'][name].to(parameter.device))
    optimizer.load_state_dict(saved)
    for name, parameter in named:
        if not torch.equal(parameter.detach().cpu(), state['parameters'][name].cpu()):
            raise ValueError('restored FP32 master differs')
    actual = optimizer.state_dict()
    if actual['param_groups'] != saved['param_groups']:
        raise ValueError('restored optimizer groups differ')
    for key in keys:
        for field, expected in saved['state'][key].items():
            if not torch.equal(actual['state'][key][field].cpu(), expected.cpu()):
                raise ValueError('restored Adam state differs')
    return dict(rank=rank, world_size=world_size, previous_updates=updates,
                master_shards=len(names), exact_masters=True, exact_adam=True,
                adapter_execution_dtype='bfloat16')
