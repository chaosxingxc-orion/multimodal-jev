import copy,json
from pathlib import Path
import pytest
import torch
from multimodal_jev.stage2_training_view import sample,classify,make_schedule,publish,load,training_inputs
from multimodal_jev.context_reference_dataset import ReferenceDataset,publish as publish_reference,digest
from multimodal_jev.baseline import file_hash
from multimodal_jev.context_topology_batch_plan import derive
from tools.reports.context_replica_formal_candidate import validate_schedule

OPTS=dict(seed=317,decision_threshold=.5,probability_margin=.1,easy_fraction=.2,min_easy_per_class=1)
BATCH=dict(training_devices=2,tensor_parallel=1,pipeline_parallel=1,per_dp_replica_batch=64,microbatch=8,inference_instances=2,inference_batch_queries=32,source_queries=10,epochs=2,efficiency_steps=100)


def rows():return [dict(id=str(i),intent='a' if i%2==0 else 'b',role='learning',official_split='train',known_pending_source=False) for i in range(20)]


def scores():
    r={x['id']:[.9,.1] if x['intent']=='a' else [.1,.9] for x in rows()}
    r['0']=[.1,.9];r['1']=[.7,.9];r['2']=[.52,.48]
    return r


def test_errors_include_top1_correct_multilabel_and_boundary():
    ids,diag,summary=sample(rows(),['a','b'],scores(),OPTS)
    d={x['id']:x for x in diag}
    assert d['0']['bucket']==d['1']['bucket']=='error'
    assert d['1']['ranking_correct'] and not d['1']['exact_set']
    assert d['2']['bucket']=='boundary'
    assert {'0','1','2'}<=set(ids) and summary['selected_easy']==2
    assert all(v['selected']>0 for v in summary['by_class'].values())


def test_order_independence_reproducibility_and_no_replacement():
    a=sample(rows(),['a','b'],scores(),OPTS)
    assert a==sample(list(reversed(rows())),['a','b'],scores(),OPTS)
    assert len(a[0])==len(set(a[0]))


@pytest.mark.parametrize('fault',['selection','test','pending','duplicate','missing','extra','nan','width'])
def test_rejects_untrusted_or_incomplete_population(fault):
    r=rows();p=scores()
    if fault=='selection':r[0]['role']='selection'
    if fault=='test':r[0]['official_split']='test'
    if fault=='pending':r[0]['known_pending_source']=True
    if fault=='duplicate':r.append(r[0])
    if fault=='missing':p.pop('0')
    if fault=='extra':p['unknown']=[.2,.8]
    if fault=='nan':p['0']=[float('nan'),.8]
    if fault=='width':p['0']=[.3,.4,.5]
    with pytest.raises(ValueError):sample(r,['a','b'],p,OPTS)


def test_tie_not_easy_and_ratio_is_not_forced_when_all_hard():
    assert classify([.5,.5],0,OPTS)['bucket']=='error'
    p={r['id']:[.8,.8] for r in rows()}
    ids,_,s=sample(rows(),['a','b'],p,OPTS)
    assert len(ids)==20 and s['actual_easy_fraction']==0


def test_schedule_covers_selected_once_per_epoch_and_zero_pads_tail():
    ids,_,_=sample(rows(),['a','b'],scores(),OPTS);c=dict(BATCH,source_queries=len(ids))
    batches=make_schedule(rows(),ids,c,317)
    validate_schedule(batches,ids,derive(c))
    assert len(batches)==2 and all(b['real_queries']==len(ids) for b in batches)
    assert all(s['weight']==0 for b in batches for s in b['slots'] if s['padding'])


def fixture(tmp_path,monkeypatch):
    from test_context_reference_dataset import fixture as original
    import multimodal_jev.local_checkpoint as ck
    monkeypatch.setattr(ck,'check_paths',lambda _:None)
    old,kw,_,_,record=original(tmp_path,monkeypatch)
    identity=copy.deepcopy(kw['expected_identity']);records=[];population={}
    for i,r in enumerate(rows()):
        rec=copy.deepcopy(record);rec['request_identity']['inputs']['input_ids']['sha256']=format(i,'064x')
        rec['request_id']=digest(rec['request_identity']);rec['prob']=torch.tensor(scores()[r['id']],dtype=torch.float32).tolist()
        records.append(rec);population[r['id']]=rec['request_id']
    identity['population_sha256']=digest(population);root=tmp_path/'full'
    result=publish_reference(root,identity=identity,records=records,population=population,producer={})
    binding=dict(root=str(root),identity=identity,population=population,manifest_sha256=result['manifest_sha256'])
    reader=ReferenceDataset(root,manifest_sha256=binding['manifest_sha256'],expected_identity=identity,expected_population=population)
    task={'schema':{'candidates':[{'id':'a'},{'id':'b'}]}}
    return reader,binding,task


def test_published_view_reuses_full_reference_and_runtime_consumes_subset(tmp_path,monkeypatch):
    reader,ref,task=fixture(tmp_path,monkeypatch);root=tmp_path/'view'
    result=publish(root,rows=rows(),task=task,reference=reader,reference_binding=ref,options=OPTS,batch_config=BATCH)
    c=dict(stage='task',training_view=dict(root=str(root),manifest_sha256=result['manifest_sha256']),reference_dataset=ref,batch_config=result['batch_plan']['configuration'],release_sha256=ref['identity']['release_sha256'],schedule_sha256=result['manifest_sha256'])
    pop,batches,m=training_inputs(c,rows(),task)
    assert len(pop)<len(ref['population']) and set(pop)<set(ref['population'])
    assert set(p.name for p in root.iterdir())=={'selection.json','batches.json','MANIFEST.json'}
    validate_schedule(batches,pop.keys(),derive(c['batch_config']))
    # Data is immutable and hard/easy assignment never leaks into actor inputs.
    (root/'batches.json').write_text('[]')
    with pytest.raises(ValueError):training_inputs(c,rows(),task)


@pytest.mark.parametrize('formal',[False,True])
def test_shared_plugin_adapter_keeps_full_cache_but_uses_view(formal):
    from tools.reports.context_training_view_trainer import generate
    source=generate(Path(__file__).resolve().parents[1],formal=formal)
    assert 'population,batches,manifest=training_inputs(c,rows,task)' in source
    assert "set(c['reference_dataset']['population'])!=set(full_population)" in source
    assert "contract['training_view']" in source
    assert 'source=full_population' in source
    compile(source,'<view>','exec')


def test_margin_alone_is_not_easy_without_threshold_anchors():
    assert classify([.51,.2],0,OPTS)['bucket']=='boundary'
    assert classify([.9,.49],0,OPTS)['bucket']=='boundary'
    assert classify([.56,.44],0,OPTS)['bucket']=='easy'


def test_reference_read_sample_is_immutable_and_missing_fails(tmp_path,monkeypatch):
    reader,_,_=fixture(tmp_path,monkeypatch)
    value=reader.read_sample('0');value['prob'][0]=.99
    assert reader.read_sample('0')['prob'][0]!=.99
    with pytest.raises((ValueError,KeyError)):reader.read_sample('missing')


def test_wrong_reference_binding_rejected_before_view_publication(tmp_path,monkeypatch):
    reader,ref,task=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='binding'):
        publish(tmp_path/'view',rows=rows(),task=task,reference=reader,
            reference_binding=dict(ref,manifest_sha256='0'*64),options=OPTS,batch_config=BATCH)
    assert not (tmp_path/'view').exists()


def test_independent_audit_reproduces_sampling(tmp_path,monkeypatch):
    import tools.reports.prepare_stage2_training_view as job
    reader,ref,task=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(job,'inputs',lambda c,p:(rows(),task,reader,ref,{'recordings':20}))
    c=dict(output=str(tmp_path/'view'),sampling=OPTS,batch_config=BATCH,audit_output=str(tmp_path/'audit.json'))
    result=job.run(c,{})
    c['manifest_sha256']=result['manifest_sha256']
    audited=job.run(c,{},audit=True)
    assert audited['recordings']==result['summary']['selected']
    assert not audited['training_dispatch_allowed']
    # Even a rehashed diagnostic tamper fails independent recomputation.
    root=Path(c['output']);data=json.loads((root/'selection.json').read_text())
    data['diagnostics'][0]['gold_probability']=.77
    (root/'selection.json').write_text(json.dumps(data))
    m=json.loads((root/'MANIFEST.json').read_text());m['files']['selection.json']=file_hash(root/'selection.json')
    (root/'MANIFEST.json').write_text(json.dumps(m));c['manifest_sha256']=file_hash(root/'MANIFEST.json')
    with pytest.raises(ValueError,match='reproducible'):job.run(c,{},audit=True)
