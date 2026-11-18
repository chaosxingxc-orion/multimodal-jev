import json
import pytest
from tools.reports.context_macro_efficiency_summary import summarize


def fixture(root):
    def write(name,v):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
    write('result.json',dict(passed=True,start_update=0,stop_update=2))
    for step in (1,2):
        receipt=dict(policy_id=str(step),head_marker_exact=True,native_loaded_tensors=112)
        write(f'activation-{step:04d}.json',dict(error=None,policy_id=str(step),
            receipt=dict(receipt,replica_receipts=[receipt,receipt])))
        rollout=dict(update=step-1,reference_misses=128 if step==1 else 0,total_seconds=4.,
            reference_seconds=1.,behavior_and_credit_seconds=2.)
        for rank in range(2):
            write(f'rank-{rank}/update-{step:04d}.json',dict(update=step,real_queries=128,next_policy_id=str(step),
                rollout=rollout,total_seconds=10.+rank,preparation_seconds=5.,actor_seconds=3.,peak_allocated=1000))


def test_complete_update_throughput_and_cold_warm_separation(tmp_path):
    fixture(tmp_path);r=summarize(tmp_path,start_update=0,stop_update=2)
    assert r['measured']['queries_per_second']==256/22
    assert r['measured']['mean_seconds']['rank0_export_checkpoint_sync_seconds']==2
    assert r['reference_miss_updates']['updates']==r['reference_hit_only_updates']['updates']==1
    r=summarize(tmp_path,start_update=0,stop_update=2,warmup_updates=1)
    assert r['all_updates']['updates']==2 and r['measured']['updates']==1


@pytest.mark.parametrize('fault',['stale','partial','extra','nested','nan','cap'])
def test_invalid_or_partial_efficiency_evidence_is_rejected(tmp_path,fault):
    fixture(tmp_path)
    p=tmp_path/'rank-1/update-0002.json';r=json.loads(p.read_text())
    if fault=='stale':r['next_policy_id']='1'
    if fault=='partial':r['real_queries']=64
    if fault=='extra':(tmp_path/'rank-0/update-0101.json').write_text('{}')
    if fault=='nested':r['preparation_seconds']=9
    if fault=='nan':r['total_seconds']=float('nan')
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError):summarize(tmp_path,start_update=0,stop_update=101 if fault=='cap' else 2)
