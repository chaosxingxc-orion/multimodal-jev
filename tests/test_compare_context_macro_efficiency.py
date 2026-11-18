"""Synthetic transport/audit substitutes test comparison, not NPU execution."""
import copy
import json
from pathlib import Path

import pytest
from tools.reports import compare_context_macro_efficiency as m


def fixture(tmp_path,monkeypatch):
    ops={};plan=dict(steps_per_arm=3,runtime_sources_sha256='',arms={})
    for micro in (8,16):
        name='micro'+str(micro);root=tmp_path/name;p=root/'package';p.mkdir(parents=True)
        remote='/remote/'+name
        t=dict(microbatch=micro,start_update=0,stop_update=3,algorithm='GRPO',
            reference_cache=remote+'/reference-cache',pins={remote+'/package/a':'hash'})
        (p/'training.json').write_text(json.dumps(t));(p/'RUNTIME-SOURCES.json').write_text('{}')
        for i in range(2):(p/f'service-{i}.json').write_text(json.dumps({'socket':remote+f'/ipc/r{i}','capacity':4}))
        plan['runtime_sources_sha256']=m.file_hash(p/'RUNTIME-SOURCES.json')
        run=root/'collected/run';run.mkdir(parents=True)
        for i in range(3):(run/f'batch-{i:04d}.json').write_text(json.dumps(dict(records=[{'id':i}],rank_indices=[[micro]],policy_id=name)))
        ops[name]=dict(prepared=str(root),operations_root=str(root),remote_root=remote,manifest_sha256=name)
        plan['arms'][name]=dict(manifest_sha256=name,remote_root=remote,microbatch=micro)
    plan_path=tmp_path/'ARMS.json';plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(m,'load_config',lambda config,local:(ops[local],{}))
    monkeypatch.setattr(m,'audit_collected',lambda c:dict(status='finite_dp2_efficiency_audited',image='same',
        files_sha256=c['manifest_sha256'],transport_sha256='hash',containers={}))
    def summary(root,**kw):
        fast='micro16' in str(root);duration=5 if fast else 10
        return dict(all_updates={'complete_seconds':duration*3},measured={'complete_seconds':duration*2},
            steps=[dict(complete_seconds=duration,reference_misses=128) for _ in range(3)])
    monkeypatch.setattr(m,'summarize',summary)
    c=dict(plan=str(plan_path),plan_sha256=m.file_hash(plan_path),arms={n:dict(config='template',local=n) for n in ops},warmup_updates=1)
    return c,ops


def test_matched_actual_inputs_and_cache_with_different_micro_partition(tmp_path,monkeypatch):
    c,_=fixture(tmp_path,monkeypatch);r=m.compare(c)
    assert r['arms']['micro16']['after_warmup_speedup']==2
    assert r['arms']['micro8']['all_updates_speedup']==1
    assert r['matched_actual_queries'] and not r['efficacy_claim']


@pytest.mark.parametrize('fault',['plan','algorithm','sample','cache','image','audit','budget','runtime','capacity'])
def test_confounded_or_unaudited_arms_refused(tmp_path,monkeypatch,fault):
    c,ops=fixture(tmp_path,monkeypatch);p=Path(ops['micro16']['prepared'])/'package'
    if fault=='plan':c['plan_sha256']='changed'
    if fault in ('algorithm','budget','capacity'):
        t=json.loads((p/'training.json').read_text());t[{'algorithm':'algorithm','budget':'stop_update','capacity':'microbatch'}[fault]]={'algorithm':'RLOO','budget':101,'capacity':32}[fault]
        (p/'training.json').write_text(json.dumps(t))
    if fault=='runtime':(p/'RUNTIME-SOURCES.json').write_text('{"changed":true}')
    if fault=='sample':(Path(ops['micro16']['operations_root'])/'collected/run/batch-0000.json').write_text('{"records":[]}')
    if fault=='cache':
        old=m.summarize
        def changed(root,**kw):
            r=old(root,**kw)
            if 'micro16' in str(root):r['steps'][0]['reference_misses']=0
            return r
        monkeypatch.setattr(m,'summarize',changed)
    if fault in ('image','audit'):
        old=m.audit_collected
        def changed(c):
            r=old(c)
            if c['manifest_sha256']=='micro16':r['image' if fault=='image' else 'status']='different'
            return r
        monkeypatch.setattr(m,'audit_collected',changed)
    with pytest.raises(ValueError):m.compare(c)
