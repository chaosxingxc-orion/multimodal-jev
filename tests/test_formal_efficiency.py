import json
import pytest
from tools.reports.context_formal_efficiency import summarize


def sample(root,step,queries):
    rollout=dict(update=step-1,total_seconds=2.,reference_seconds=0.,behavior_and_credit_seconds=1.)
    for rank in range(2):
        p=root/f'rank-{rank}';p.mkdir(exist_ok=True)
        (p/f'update-{step:04d}.json').write_text(json.dumps(dict(update=step,real_queries=queries,next_policy_id='p',
            rollout=rollout,total_seconds=5.,preparation_seconds=2.,actor_seconds=2.,peak_allocated=123)))
    receipt=dict(policy_id='p',head_marker_exact=True,native_loaded_tensors=112)
    (root/f'activation-{step:04d}.json').write_text(json.dumps(dict(error=None,policy_id='p',
        receipt=dict(receipt,replica_receipts=[receipt,receipt]))))


def test_formal_tail_uses_real_denominator_and_windows_after_100(tmp_path):
    sample(tmp_path,316,128);sample(tmp_path,317,65);sample(tmp_path,318,128)
    r=summarize(tmp_path,start_update=315,stop_update=318,warmup_updates=1)
    assert r['real_query_exposures']==193 and r['queries_per_second']==19.3
    assert not r['separate_training']


def test_partial_rank_or_ack_is_not_committed_sample(tmp_path):
    sample(tmp_path,1,128);(tmp_path/'rank-1/update-0001.json').unlink()
    with pytest.raises(FileNotFoundError):summarize(tmp_path,start_update=0,stop_update=1,warmup_updates=0)


def test_nonfinite_timing_rejected(tmp_path):
    sample(tmp_path,1,128);p=tmp_path/'rank-0/update-0001.json';v=json.loads(p.read_text());v['total_seconds']=float('nan');p.write_text(json.dumps(v))
    with pytest.raises(ValueError,match='finite'):summarize(tmp_path,start_update=0,stop_update=1,warmup_updates=0)
