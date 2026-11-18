import copy

import pytest

from multimodal_jev.span_batched_training_metrics import aggregate_update


def records():
    return [dict(update=317, batch_layout=dict(world_size=4,global_batch=128,accumulation_per_rank=4),
        admission=dict(stage='instruction',global_real_requests=65),
        metrics={'jev/ce_contribution':[1.,2.,3.,4.], 'jev/rl_contribution':[0.]*4},
        seconds=10.+rank,total_step_seconds=15.+rank,peak_memory_allocated=100+rank,
        peak_memory_reserved=200+rank) for rank in range(4)]


def test_sum_micros_mean_ranks_and_real_tail_denominator():
    summary=aggregate_update(records())
    assert summary['weighted_ce_real_request_mean']==10.
    assert summary['update_requests_per_second']==65/13
    assert summary['total_step_requests_per_second']==65/18
    assert summary['peak_allocated_bytes']==103


@pytest.mark.parametrize('fault',['missing_rank','update','population','micro','nan','rl'])
def test_partial_or_misaligned_metrics_rejected(fault):
    rows=copy.deepcopy(records())
    if fault=='missing_rank':rows.pop()
    if fault=='update':rows[0]['update']=318
    if fault=='population':rows[0]['admission']['global_real_requests']=128
    if fault=='micro':rows[0]['metrics']['jev/ce_contribution'].pop()
    if fault=='nan':rows[0]['metrics']['jev/ce_contribution'][0]=float('nan')
    if fault=='rl':rows[0]['metrics']['jev/rl_contribution'][0]=1.
    with pytest.raises(ValueError):aggregate_update(rows)
