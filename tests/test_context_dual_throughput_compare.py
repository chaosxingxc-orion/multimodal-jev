"""Synthetic collected evidence checks; not a measured speedup report."""
import json
from pathlib import Path

import pytest
import torch

from tools.reports.prepare_context_dual_throughput_pair import write,sha
from tools.reports.compare_context_dual_throughput_pair import compare,summarize


def fixture(tmp_path):
    c={};arms={}
    for name,capacity,seconds in (('serial',1,10.),('batch4',4,5.)):
        root=tmp_path/name;(root/'supervision').mkdir(parents=True);(root/'service').mkdir()
        remote='/remote/'+name
        write(root/'config.json',dict(start_update=0,stop_update=2,checkpoint_interval=1,
            engineering_prefix=True,training_dispatch_allowed=False,reference_cache=remote+'/cache'))
        write(root/'service-config.json',dict(policy=dict(max_batch_queries=capacity,max_batch_tokens=16384)))
        write(root/'PREPARED.json',dict(remote_root=remote))
        write(root/'supervision/terminal.json',dict(passed=True,forced_cleanup=False,
            actor_exit_code=0,service_exit_code=0,child_exit_codes=[0,0]))
        write(root/'service/terminal.json',dict(exit_code=0));(root/'run').mkdir()
        write(root/'run/result.json',dict(passed=True))
        for step in (1,2):
            shards={}
            for rank in range(4):
                dest=root/f'run/rank-{rank}';dest.mkdir(exist_ok=True)
                write(dest/f'update-{step:04d}.json',dict(update=step,checkpoint=True,real_queries=128,
                    total_seconds=seconds+rank/10,preparation_seconds=1.,actor_seconds=1.,
                    behavior_parity_history=[[dict(passed=True) for _ in range(4)] for _ in range(4)],
                    next_policy_id=f'p{step}',rollout=dict(reference_misses=128)))
                p=dest/f'checkpoint-{step:04d}.pt'
                torch.save(dict(parameters={'head':torch.ones(2)*step},optimizer={'state':torch.ones(2)*rank}),p)
                shards[str(p.relative_to(root/'run'))]=sha(p)
            write(root/f'run/checkpoint-{step:04d}-committed.json',dict(rank_shards=shards))
        wire=dict(status='transport_complete_independent_audit_required',
            state=dict(Status='exited',ExitCode=0,OOMKilled=False),bad_pins=[],bad_package=[],
            launch_binding_mismatches=[],image='fixed',container_id=name,
            files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()})
        transport=tmp_path/(name+'-transport.json');write(transport,wire)
        c[name]=dict(collection=str(root),transport=str(transport))
        arms[name]=dict(manifest_sha256=sha(root/'PREPARED.json'),max_batch_queries=capacity)
    p=tmp_path/'PAIR.json';write(p,dict(arms=arms))
    c.update(pair=str(p),pair_sha256=sha(p));return c


def reseal(c,name):
    root=Path(c[name]['collection']);p=Path(c[name]['transport']);w=json.loads(p.read_text())
    w['files']={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()}
    p.write_text(json.dumps(w))


def test_full_step_timing_uses_slowest_rank_and_never_authorizes_switch(tmp_path):
    c=fixture(tmp_path);r=compare(c)
    assert r['serial_seconds']==20.6 and r['batch4_seconds']==10.6
    assert r['exact_two_step_checkpoints'] and not r['switch_old_training_allowed']


@pytest.mark.parametrize('fault',['missing_rank','timer','cache','parity','optimizer'])
def test_invalid_or_changed_evidence_is_rejected(tmp_path,fault):
    c=fixture(tmp_path);root=Path(c['batch4']['collection'])
    p=root/'run/rank-3/update-0002.json';d=json.loads(p.read_text())
    if fault=='missing_rank':p.unlink()
    elif fault=='optimizer':
        q=root/'run/rank-3/checkpoint-0002.pt';torch.save({'changed':torch.zeros(1)},q)
        commit=root/'run/checkpoint-0002-committed.json';v=json.loads(commit.read_text())
        v['rank_shards']['rank-3/checkpoint-0002.pt']=sha(q);commit.write_text(json.dumps(v))
    else:
        if fault=='timer':d['total_seconds']=float('nan')
        if fault=='cache':
            p=root/'run/rank-0/update-0002.json';d=json.loads(p.read_text());d['rollout']['reference_misses']=0
        if fault=='parity':d['behavior_parity_history'][0][0]['passed']=False
        p.write_text(json.dumps(d))
    reseal(c,'batch4')
    with pytest.raises((ValueError,FileNotFoundError)):compare(c)
