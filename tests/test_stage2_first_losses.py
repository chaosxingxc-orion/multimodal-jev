import json
import pytest
from tools.reports.watch_stage2_first_losses import first_losses


def fixture(root):
    for step in range(1,4):
        policy=f'p{step}';receipt=dict(policy_id=policy,head_marker_exact=True,native_loaded_tensors=112)
        (root/f'activation-{step:04d}.json').write_text(json.dumps(dict(error=None,policy_id=policy,
            receipt=dict(receipt,replica_receipts=[receipt,receipt]))))
        for rank in range(2):
            d=root/f'rank-{rank}';d.mkdir(exist_ok=True)
            metrics={'actor/pg_loss':3.,'jev/context_loss':[1.,2.],'jev/dual_rl':[0.,0.],
                     'jev/dual_auxiliary':[1.,1.],'jev/dual_kl_unweighted':[50.,50.]}
            (d/f'update-{step:04d}.json').write_text(json.dumps(dict(update=step,next_policy_id=policy,
                rollout=dict(behavior_policy_id=f'p{step-1}'),real_queries=128,metrics=metrics)))


def test_sum_microbatches_then_average_ranks(tmp_path):
    fixture(tmp_path);rows=first_losses(tmp_path)
    assert len(rows)==3 and rows[0]['losses']['jev/context_loss']==3.


def test_missing_ack_or_inconsistent_actor_loss_fails(tmp_path):
    fixture(tmp_path);p=tmp_path/'rank-1/update-0002.json';v=json.loads(p.read_text())
    v['metrics']['actor/pg_loss']=5.;p.write_text(json.dumps(v))
    with pytest.raises(ValueError,match='actor loss'):first_losses(tmp_path)
    p.unlink()
    with pytest.raises(FileNotFoundError):first_losses(tmp_path)
