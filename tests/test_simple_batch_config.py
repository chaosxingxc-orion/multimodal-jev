import pytest
from multimodal_jev.simple_batch_config import batch_layout
from multimodal_jev.slurp_instruction_schedule import epoch_schedule


@pytest.mark.parametrize('size', [128, 256])
@pytest.mark.parametrize('micro', [1, 2, 4])
def test_batch_arithmetic(size, micro):
    layout = batch_layout(dict(global_batch=size, microbatch_per_rank=micro))
    assert 4 * micro * layout['accumulation_per_rank'] == size
    assert layout['local_batch'] == size // 4


@pytest.mark.parametrize('config', [dict(global_batch=True), dict(global_batch=64),
    dict(global_batch=256, microbatch_per_rank=3), dict(global_batch=128, microbatch_per_rank=True)])
def test_invalid_batch_rejected(config):
    with pytest.raises(ValueError):
        batch_layout(config)


def test_larger_batch_preserves_complete_learning_order_weights_and_exposure():
    rows = [dict(id=str(i), source_group=str(i//3), intent=str(i%60),
                 role='learning', known_pending_source=False) for i in range(40513)]
    for epoch in (0, 1):
        schedules = [epoch_schedule(rows, seed=317, epoch=epoch, global_batch=size,
                     task_mode='direct_final', class_weighting='sqrt_cap4') for size in (128,256)]
        real = [[s for b in batches for s in b if not s['padding']] for batches,_ in schedules]
        assert real[0] == real[1] and len(real[1]) == 40513
        assert len({s['id'] for s in real[1]}) == 40513
        assert [r['updates'] for _,r in schedules] == [317,159]
        assert [r['padding_slots'] for _,r in schedules] == [63,191]
        assert all(s['weight']==0 for batches,_ in schedules for b in batches for s in b if s['padding'])
