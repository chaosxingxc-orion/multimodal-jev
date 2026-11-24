import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from multimodal_jev import window_recovery as recovery, evaluation_blocks as blocks
from multimodal_jev.baseline import file_hash
from multimodal_jev.context_execution import content_key
from tools.reports.score_probability_panels import compare_history


class OneRank:
    def broadcast_object_list(self, value, src): pass
    def all_gather_object(self, values, local): values[0] = local
    def barrier(self): pass


def test_last_committed_generation_identical_next_adam_and_tamper(tmp_path,monkeypatch):
    monkeypatch.setattr('multimodal_jev.local_checkpoint.check_paths',lambda _:None)
    p = torch.nn.Parameter(torch.tensor([0.5, -1.25]))
    def opt(): return torch.optim.AdamW([p],lr=.01)
    def step(o):
        o.zero_grad(); p.square().sum().backward(); o.step()
    o = opt()
    named = [('head',p)]
    contract = {'target':'frozen'}
    for i in (1,2):
        step(o)
        identity=dict(rank=0,world_size=1,completed_updates=i,contract=contract)
        recovery.save(tmp_path,named,o,identity,{'cursor':i},OneRank())
    # A partial later generation cannot advance the published cursor.
    (tmp_path/'recovery/generation-incomplete').mkdir()
    proof=recovery.latest(tmp_path,contract,0,4,1)
    assert proof['step']==2 and len(list((tmp_path/'recovery').glob('step-*.json')))==1
    with pytest.raises(ValueError):
        recovery.latest(tmp_path,{'target':'another'},0,4,1)
    step(o); expected=p.detach().clone()
    other=opt()
    assert recovery.recover(tmp_path,proof,named,other,identity)=={'cursor':2}
    step(other)
    assert torch.equal(p,expected)
    for key,value in o.state[p].items(): assert torch.equal(value,other.state[p][key])
    (tmp_path/'recovery'/proof['generation']/'rank-0.pt').write_bytes(b'truncated')
    with pytest.raises(ValueError,match='shard digest'):
        recovery.recover(tmp_path,proof,named,opt(),identity)


def test_history_paired_same_arm_and_changed_archive_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr('multimodal_jev.local_checkpoint.check_paths',lambda _:None)
    def put(path,obj):
        path.parent.mkdir(exist_ok=True,parents=True); path.write_text(json.dumps(obj))
        return file_hash(path)
    ds='demo'; ids=['x','y']
    rows=[dict(id=str(i),audio_sha256='audio'+str(i),gold=ids[i%2]) for i in range(4)]
    rel='scoring/demo/targets.json'
    seal=put(tmp_path/'MANIFEST.json',dict(files={rel:put(tmp_path/rel,rows)}))
    cache=dict(dataset=ds,seal=dict(root=str(tmp_path),sha256=seal))
    jobs=[(ds,r,SimpleNamespace(questions=[SimpleNamespace(id=i) for i in ids])) for r in rows]
    arms=('stage1','tpo')
    plan=dict(schema=blocks.SCHEMA,namespace='history',choice=False,policies={a:'209-'+a for a in arms},
              inputs={ds:dict(seal_sha256=seal)},blocks=[dict(dataset=ds,arm=a,key=f'block-{j:06d}',
              ids=[r['id'] for r in rows],widths=[2]*4) for j,a in enumerate(arms)])
    manifest=dict(plan=plan,sha256=content_key(plan));put(tmp_path/'plan.json',manifest)
    preds={}
    scores={}
    for j,b in enumerate(plan['blocks']):
        arm=b['arm']
        old=[dict(id=r['id'],prob=[float((i+j)%2==k) for k in range(2)]) for i,r in enumerate(rows)]
        blocks.publish(tmp_path,manifest,b,{'prob':dict(policy_id=plan['policies'][arm],records=old)})
        preds[arm]=({r['id']:dict(prob=r['prob'][::-1]) for r in old},'210-'+arm)
        scores[arm]=dict(policy_id=plan['policies'][arm],top1_macro_f1=1-j,top1_accuracy=1-j,queries=4)
    metrics=tmp_path/'metrics.json'
    spec=dict(archive_root=str(tmp_path),plan_digest=manifest['sha256'],experiment='JEV-209',
              metrics=dict(path=str(metrics),sha256=put(metrics,dict(seals={ds:seal},datasets={ds:dict(arms=scores)}))))
    protocol=dict(cluster='id',interval=dict(method='paired_cluster_percentile_bootstrap',confidence=.95,samples=100,seed=7))
    proof=compare_history(spec,cache,jobs,preds,protocol,2)
    assert [proof[a]['direction'] for a in arms]==['退化','改善']
    assert [proof[a]['macro_f1_delta'] for a in arms]==[-1,1]
    (tmp_path/'block-000000.json').write_text((tmp_path/'block-000000.json').read_text().replace('209-stage1','alien'))
    with pytest.raises(ValueError,match='content differs'):
        compare_history(spec,cache,jobs,preds,protocol,2)


def test_registered_stage2_view_reuse_requires_actual_score_and_each_file(tmp_path,monkeypatch):
    from tools.reports import experiment_package as driver
    from multimodal_jev.local_config import TOKEN
    spec=json.loads(Path('experiments/pipelines/JEV-210.json').read_bytes())
    st=next(s for s in spec['stages']['train'] if s['id']=='build-fused-stage2-view')
    config=Path(st['config']).read_bytes()
    monkeypatch.setattr(driver,'committed',lambda *_:config)
    macros={k:str(tmp_path/k) for k in TOKEN.findall(config.decode()+json.dumps(st))}
    report=Path(macros['CARD_ENDPOINT_SCORE'])/'audit.json'
    report.parent.mkdir(); report.write_text(json.dumps(dict(labels=['a','b'],checkpoint_sha256='NEW',
                   policy_id='pid',policy_sha256='policy',full_sha256='roster')))
    root=Path(macros['CARD_TPO_ROOT'])/'targets'; root.mkdir(parents=True)
    manifest=dict(schema='jev-tpo-reward-view-v1',card_input=dict(checkpoint_sha256='OLD'),candidate_ids=['a','b'],
        schedule=dict(global_batch=384,epochs=1),probability_gap=dict(gate='strict_target_top1'),
        reference_sha256=file_hash(report),view_sha256='roster',source_policy='pid',source_policy_sha256='policy',
        recordings=2,updates=1,strata={},files={})
    for k in st['checks'][0]['verify_files']:
        (root/k).write_bytes(k.encode())
        manifest['files'][k]=file_hash(root/k)
    def run():
        (root/'MANIFEST.json').write_text(json.dumps(manifest))
        return driver.step_result(st,macros,tmp_path,{'commit':'test'},None,[])
    with pytest.raises(ValueError,match='does not match prerequisite'): run()
    manifest['card_input']['checkpoint_sha256']='NEW'
    assert run()['mode']=='reused'
    (root/'reference.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='file hash differs'): run()


def test_published_design_and_legacy_cache_rebinding(tmp_path,monkeypatch):
    from tools.reports import prepare_card_stage1 as prepare
    cfg=json.loads(Path('configs/portable/operations/card-stage1-knegative-ema-dp8-v9.json').read_bytes())
    monkeypatch.setattr('multimodal_jev.local_checkpoint.check_paths',lambda _:None)
    cfg.update(output=str(tmp_path/'out'))
    cfg['mixed']['data']=str(tmp_path)
    cfg['mixed']['protocol']=str(tmp_path/'protocol.json')
    for name in ['protocol','synthetic','admission','annotation-errors']:
        (tmp_path/(name+'.json')).write_text(json.dumps({'extra_cards':[]}))
    old_load=prepare.load
    def load(p,h):
        if Path(p).name=='admission.json': return dict(population=109610,roster_sha256='dummy')
        if Path(p).name=='synthetic.json': return []
        if p==cfg['parent']: raise StopIteration
        return old_load(p,h)
    monkeypatch.setattr(prepare,'load',load)
    def compile_plan(c,entry,design_sha,extension):
        prepare.commit_json(entry/'card-plan.json',dict(design_sha256=design_sha,assignments={'audio':['a','b']}))
    monkeypatch.setattr(prepare,'compile_plan',compile_plan)
    import copy
    old=copy.deepcopy(cfg);old.pop('bind_design_budget');old['output']=str(tmp_path/'old')
    with pytest.raises(StopIteration): prepare.run(old)
    with pytest.raises(StopIteration): prepare.run(cfg)
    d=json.loads((tmp_path/'out/entry/design.json').read_bytes())['stage1']
    assert [d[k] for k in ('global_batch','updates_per_epoch','total_updates','tail_real','tail_padding')]==[64,1713,3426,42,22]
    assert cfg['window']['cache_card_plan_sha256']==file_hash(tmp_path/'old/entry/card-plan.json')
    assert d['evaluation_steps']==cfg['window']['evaluation_steps']==[0,1713,3426]
