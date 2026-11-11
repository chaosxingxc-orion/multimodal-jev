import copy

import pytest

torch = pytest.importorskip('torch')
from multimodal_jev.span_fsdp_resume import restore_local_shard, validate_cursor


def fixture():
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 2, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    model(torch.ones(1, 3)).square().sum().backward()
    optimizer.step()
    state = dict(rank=0, world_size=4, updates=1,
                 names=[n for n, _ in model.named_parameters()],
                 parameters={n: p.detach().clone() for n, p in model.named_parameters()},
                 optimizer=copy.deepcopy(optimizer.state_dict()))
    loaded = copy.deepcopy(model)
    for p in loaded.parameters():
        p.data = p.data.to(torch.bfloat16).float()
    fresh = torch.optim.AdamW(loaded.parameters(), lr=1e-4, foreach=False)
    return model, optimizer, loaded, fresh, state


def restore(model, optimizer, state):
    return restore_local_shard(model, optimizer, state, rank=0, world_size=4, updates=1)


def test_restore_preserves_sub_bf16_masters_and_next_adam_step():
    model, optimizer, loaded, fresh, state = fixture()
    assert not torch.equal(model.weight, loaded.weight)
    assert restore(loaded, fresh, state)['exact_adam']
    for m, opt in [(model, optimizer), (loaded, fresh)]:
        opt.zero_grad()
        m(torch.ones(1, 3)).square().sum().backward()
        opt.step()
    assert torch.equal(model.weight, loaded.weight)
    for field, value in optimizer.state_dict()['state'][0].items():
        assert torch.equal(value, fresh.state_dict()['state'][0][field])


@pytest.mark.parametrize('damage', [
    'rank', 'world_size', 'updates', 'names', 'master_dtype', 'master_shape',
    'master_policy', 'missing_adam', 'step', 'moment_nan', 'moment_shape',
    'moment_dtype', 'negative_second', 'lr',
])
def test_corrupt_state_rejected_before_parameter_mutation(damage):
    _, _, model, optimizer, state = fixture()
    if damage in ('rank', 'world_size', 'updates'):
        state[damage] += 1
    elif damage == 'names': state['names'] = ['wrong']
    elif damage == 'master_dtype': state['parameters']['weight'] = state['parameters']['weight'].bfloat16()
    elif damage == 'master_shape': state['parameters']['weight'] = torch.ones(6)
    elif damage == 'master_policy': state['parameters']['weight'] += 1
    elif damage == 'missing_adam': state['optimizer']['state'] = {}
    elif damage == 'step': state['optimizer']['state'][0]['step'] += 1
    elif damage == 'moment_nan': state['optimizer']['state'][0]['exp_avg'][0, 0] = float('nan')
    elif damage == 'moment_shape': state['optimizer']['state'][0]['exp_avg'] = torch.zeros(6)
    elif damage == 'moment_dtype': state['optimizer']['state'][0]['exp_avg'] = torch.zeros(2, 3).bfloat16()
    elif damage == 'negative_second': state['optimizer']['state'][0]['exp_avg_sq'][0, 0] = -1
    elif damage == 'lr': state['optimizer']['param_groups'][0]['lr'] *= 2
    before = model.weight.detach().clone()
    with pytest.raises(ValueError): restore(model, optimizer, state)
    assert torch.equal(before, model.weight)
    assert not optimizer.state


def test_rank_cursor_advances_by_global_batch_not_one_pair():
    previous = dict(schema='jev-span-balanced-cursor-v1', seed=17,
                    schedule_sha256='schedule', epoch_batches=20,
                    manifest_sha256='manifest', question_order=['a', 'b', 'c'],
                    batch_index=2)
    current = dict(previous, batch_index=6)
    validate_cursor(previous, current, 4)
    with pytest.raises(ValueError): validate_cursor(previous, dict(current, batch_index=3), 4)
    with pytest.raises(ValueError): validate_cursor(previous, dict(current, seed=29), 4)
    with pytest.raises(ValueError): validate_cursor(dict(previous, batch_index=18), dict(current, batch_index=22), 4)
