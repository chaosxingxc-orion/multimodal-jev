import copy
import pytest
from multimodal_jev.simple_batch_config import batch_layout
from multimodal_jev.simple_throughput import summarize


def fixture():
    config=dict(global_batch=128,microbatch_per_rank=1,updates=2,frozen_audit='checkpoint')
    ranks=[]
    for rank in range(4):
        steps=[]
        for update in (1,2):
            row=dict(update=update,batch_layout=batch_layout(config),seconds=10.,preparation_seconds=2.,
                     audit_seconds=1.,total_step_seconds=14.+rank,peak_memory_allocated=100,
                     peak_memory_reserved=200,frozen_audit='checkpoint',frozen_hash_checked=update==2,
                     admission=dict(global_real_requests=128 if update==1 else 65,
                                    global_rows=128,reduction='real_request_mean'))
            if update==2:row.update(checkpoint_seconds=1.,adapter_id='end')
            steps.append(row)
        ranks.append(dict(rank=rank,passed=True,final_adapter_id='end',step_metrics=steps))
    return config,ranks


def test_real_counts_padding_and_parallel_rank_time():
    c,r=fixture();s=summarize(c,r)
    assert s['real_requests']==193 and s['measured_step_seconds']==34
    assert s['real_requests_per_second']==pytest.approx(193/34)
    assert s['steps'][1]['real_requests_per_second']==pytest.approx(65/17)
    assert s['maximum_reserved_bytes']==200


@pytest.mark.parametrize('fault',['missing_rank','duplicate_rank','partial_step','duplicate_step','policy',
    'denominator','batch','timing_nan','timing_negative','timing_incomplete','missing_audit','memory'])
def test_incomplete_or_inconsistent_measurements_rejected(fault):
    c,r=fixture();s=r[0]['step_metrics'][1]
    if fault=='missing_rank':r.pop()
    elif fault=='duplicate_rank':r[0]['rank']=1
    elif fault=='partial_step':r[0]['step_metrics'].pop()
    elif fault=='duplicate_step':s['update']=1
    elif fault=='policy':r[0]['final_adapter_id']='other'
    elif fault=='denominator':s['admission']['global_real_requests']=64
    elif fault=='batch':s['batch_layout']['microbatch_per_rank']=2
    elif fault=='timing_nan':s['seconds']=float('nan')
    elif fault=='timing_negative':s['checkpoint_seconds']=-1.
    elif fault=='timing_incomplete':s['total_step_seconds']=10.
    elif fault=='missing_audit':s['frozen_hash_checked']=False
    else:s['peak_memory_reserved']=50
    with pytest.raises(ValueError):summarize(c,r)
