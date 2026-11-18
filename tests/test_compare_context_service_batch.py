"""Actual100-step timer aggregation; runtime audit receipts are test fixtures."""
import json
from pathlib import Path

import pytest

from tools.reports.compare_context_service_batch import compare


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path,ratio=2.):
    operations=[];audits=[]
    for name,batch,tokens,duration in [('old',4,16384,10.),('new',32,131072,10./ratio)]:
        root=tmp_path/name;package=root/'package';run=root/'collected/run';remote='/remote/'+name
        training=dict(microbatch=8,global_batch=128,start_update=0,stop_update=100,
            experiment_kind='efficiency',algorithm='joint_grpo',source_policy_id='initial',
            reference_cache=remote+'/reference-cache',pins={remote+'/package/code.py':'fixed'})
        write(package/'training.json',training)
        for rank in range(2):
            write(package/f'service-{rank}.json',dict(policy=dict(max_batch_queries=batch,
                max_batch_tokens=tokens,policy_id='initial'),socket=remote+f'/ipc/{rank}'))
        write(run/'result.json',dict(passed=True,start_update=0,stop_update=100))
        for step in range(1,101):
            policy=f'policy-{step}'
            ack=dict(policy_id=policy,head_marker_exact=True,native_loaded_tensors=112)
            write(run/f'activation-{step:04d}.json',dict(error=None,policy_id=policy,
                receipt=dict(ack,replica_receipts=[ack,ack])))
            rollout=dict(update=step-1,reference_misses=128,total_seconds=duration*.4,
                reference_seconds=duration*.15,behavior_and_credit_seconds=duration*.2)
            for rank in range(2):
                write(run/f'rank-{rank}/update-{step:04d}.json',dict(update=step,
                    real_queries=128,next_policy_id=policy,rollout=rollout,
                    total_seconds=duration,preparation_seconds=duration*.5,
                    actor_seconds=duration*.4,peak_allocated=1000))
            write(run/f'batch-{step-1:04d}.json',dict(records=[{'id':step,'weight':1.}]))
        operations.append(dict(prepared=str(root),operations_root=str(root),remote_root=remote,manifest_sha256=name))
        audits.append(dict(status='finite_dp2_efficiency_audited',files_sha256=name,image='fixed'))
    return operations,audits


@pytest.mark.parametrize('ratio',[2.,.8])
def test_measured_speedup_or_regression_retained_across_full_and_warm_windows(tmp_path,ratio):
    ops,audits=fixture(tmp_path,ratio)
    r=compare(*ops,audits)
    assert r['all_updates_speedup']==pytest.approx(ratio)
    assert r['after_warmup_speedup']==pytest.approx(ratio)
    assert r['saved_complete_seconds']==pytest.approx(1000-1000/ratio)
    assert r['candidate']['measured']['updates']==98
    assert r['candidate']['all_updates']['real_query_exposures']==12800
    assert r['candidate']['all_updates']['queries_per_second']==pytest.approx(12.8*ratio)


@pytest.mark.parametrize('fault',['micro','algorithm','budget','capacity','token_budget','sample',
                                 'cache','missing_update','extra_update','image','audit','manifest'])
def test_no_confounded_partial_or_unaudited_comparison(tmp_path,fault):
    ops,audits=fixture(tmp_path);root=Path(ops[1]['prepared']);run=root/'collected/run'
    if fault in ('micro','algorithm','budget'):
        p=root/'package/training.json';c=json.loads(p.read_text())
        key,value={'micro':('microbatch',16),'algorithm':('algorithm','changed'),'budget':('stop_update',99)}[fault]
        c[key]=value;write(p,c)
    if fault in ('capacity','token_budget'):
        p=root/'package/service-0.json';c=json.loads(p.read_text())
        c['policy']['max_batch_queries' if fault=='capacity' else 'max_batch_tokens']=48
        write(p,c)
    if fault=='sample':write(run/'batch-0099.json',dict(records=[]))
    if fault=='cache':
        for rank in range(2):
            p=run/f'rank-{rank}/update-0005.json';c=json.loads(p.read_text())
            c['rollout']['reference_misses']=0;write(p,c)
    if fault=='missing_update':(run/'rank-1/update-0100.json').unlink()
    if fault=='extra_update':write(run/'rank-0/update-0101.json',{})
    if fault in ('image','audit','manifest'):
        audits[1][{'image':'image','audit':'status','manifest':'files_sha256'}[fault]]='wrong'
    with pytest.raises(ValueError):compare(*ops,audits)
