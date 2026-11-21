import copy
import json
from pathlib import Path
import pytest
import torch
from multimodal_jev import stage2_dataset_release as module
from multimodal_jev.baseline import file_hash
from multimodal_jev.context_reference_dataset import digest
from multimodal_jev.stage2_training_view import sample
from test_stage2_training_view import fixture,rows,scores,OPTS,BATCH

EXEC=dict(io_workers=2,cpu_workers=1,min_parallel=1,chunk=4,window=2)
BATCH_CONFIG={k:v for k,v in BATCH.items() if k!='source_queries'}
OBJ=dict(kind='candidate_infonce_probability_margin_v1',temperature=1.,probability_coefficient=0.,
    probability_margin=dict(margin=.1,decision_threshold=.5,gap_weight=1.,anchor_weight=.25))


def setup(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'check_paths',lambda c:None)
    reader,ref,task=fixture(tmp_path,monkeypatch)
    task['schema']['candidates']=[dict(id='a',description='first'),dict(id='b',description='second')]
    task['thinking']=dict(requirements=['Listen to audio'])
    population=[dict(r,audio='/audio/'+r['id']+'.wav',audio_sha256='a'*64) for r in rows()]
    batch=dict(BATCH_CONFIG,training_devices=4,inference_instances=0,source_queries=len(population))
    endpoint=module.derive(batch)['total_updates']
    checkpoint=dict(updates=endpoint,joint_policy={'policy_id':reader.policy_id},contract=dict(stage='instruction',world_size=4,
        batch_config=batch,objective=OBJ,release_sha256=ref['identity']['release_sha256'],
        base_id=ref['identity']['base_id'],source_contract=ref['identity']['contract'],prompt_layout='prefix-v1'))
    expected=dict(policy_id=reader.policy_id,stage1_commit_sha256='f'*64,endpoint=endpoint)
    bound=module.endpoint_binding(checkpoint,commit_sha256='f'*64,reference=reader,source_expected=expected,sampling=OPTS)
    args=dict(rows=population,task=task,reference=reader,endpoint=bound,options=OPTS,
        batch_config=BATCH_CONFIG,evaluation={'interval_updates':64},execution=EXEC)
    return args,checkpoint,expected,ref


def test_publish_reaudit_reuse_and_dynamic_budget(tmp_path,monkeypatch):
    a,_,_,ref=setup(tmp_path,monkeypatch);root=tmp_path/'view'
    result=module.publish(root,**a);again=module.publish(root,**a)
    assert not result['reused'] and again['reused']
    assert result['manifest_sha256']==again['manifest_sha256']
    assert result['consumer']['stop_update']==2
    assert result['consumer']['evaluation_steps']==[0,2]
    assert result['consumer']['batch_config']['source_queries']==5
    assert result['consumer']['real_exposures']==10
    assert {p.name for p in root.iterdir()}=={'MANIFEST.json',*module.FILES}
    dataset=module.TrainingDataset(root,result['manifest_sha256'],rows=a['rows'],task=a['task'],reference=a['reference'],execution=EXEC)
    config=dict(stage='task',start_update=0,source_policy_id=a['reference'].policy_id,release_sha256=ref['identity']['release_sha256'],reference_dataset=ref,
        stop_update=634,batch_config=BATCH,global_batch=128,macro_client={'max_queries':128,'rollout_queries':32},
        objective_plugin={'options':{'custom_reward':True}},parent_schedule='/old',checkpoint_interval=64)
    new=dataset.bind(config)
    assert new['stop_update']==2 and 'parent_schedule' not in new and not new['training_dispatch_allowed']
    assert config['stop_update']==634 and new['objective_plugin']==config['objective_plugin']
    assert new['reference_dataset']==ref
    assert set(dataset.population)==set(json.loads((root/'selection.json').read_text())['selected_ids'])


def test_existing_batch_consumer_gets_complete_candidates_no_mining_metadata(tmp_path,monkeypatch):
    from multimodal_jev.cached_context_preparation import PreparedContextInput
    from multimodal_jev.encoded_audio_cache import canonical
    a,_,_,_=setup(tmp_path,monkeypatch);root=tmp_path/'view';result=module.publish(root,**a)
    dataset=module.TrainingDataset(root,result['manifest_sha256'],rows=a['rows'],task=a['task'],reference=a['reference'],execution=EXEC)
    seen=[]
    def prepare(job):
        request,sha=job;seen.append(request.model_input())
        return PreparedContextInput(canonical(request.model_input()),dict(input_ids=torch.tensor([[99,99]])),torch.tensor([[0,1]]),{},True)
    with dataset.stream(prepare) as stream:
        for step in range(2):
            items,metadata=stream.consume(step,'actor-new')
            assert len(items)==128 and sum(x['real_request'] for x in items)==5
            assert all(x['labels'].shape==(1,2) for x in items)
            assert {m['id'] for m,x in zip(metadata,items) if x['real_request']}==set(dataset.population)
            assert all(x['request_weight']==float(x['real_request']) for x in items)
    assert all('bucket' not in json.dumps(v) and 'gold_probability' not in json.dumps(v) for v in seen)
    assert len(dataset.reference.population)==20


@pytest.mark.parametrize('fault',['policy','commit','prefix','threshold','margin','stage','release'])
def test_wrong_endpoint_or_threshold_rejected(tmp_path,monkeypatch,fault):
    a,checkpoint,expected,_=setup(tmp_path,monkeypatch);options=copy.deepcopy(OPTS)
    if fault=='policy':expected['policy_id']='0'*64
    if fault=='commit':expected['stage1_commit_sha256']='0'*64
    if fault=='prefix':checkpoint['updates']=expected['endpoint']=1
    if fault=='threshold':options['decision_threshold']=.6
    if fault=='margin':options['probability_margin']=.2
    if fault=='stage':checkpoint['contract']['stage']='task'
    if fault=='release':checkpoint['contract']['release_sha256']='0'*64
    with pytest.raises(ValueError):module.endpoint_binding(checkpoint,commit_sha256='f'*64,reference=a['reference'],source_expected=expected,sampling=options)


def test_rehashed_sampling_tamper_rejected(tmp_path,monkeypatch):
    a,_,_,_=setup(tmp_path,monkeypatch);root=tmp_path/'view';module.publish(root,**a)
    payload=json.loads((root/'selection.json').read_text());payload['selected_ids'].reverse()
    (root/'selection.json').write_text(json.dumps(payload))
    manifest=json.loads((root/'MANIFEST.json').read_text());manifest['files']['selection.json']=file_hash(root/'selection.json')
    (root/'MANIFEST.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='reproducible'):module.audit(root,file_hash(root/'MANIFEST.json'),**a)


def test_reuse_rejects_changed_seed_code_or_task(tmp_path,monkeypatch):
    a,_,_,_=setup(tmp_path,monkeypatch);root=tmp_path/'view';module.publish(root,**a)
    with pytest.raises(ValueError,match='identity'):module.publish(root,**dict(a,options=dict(OPTS,seed=999)))
    changed=copy.deepcopy(a['task']);changed['thinking']['requirements']=['Other rule']
    with pytest.raises(ValueError,match='identity'):module.publish(root,**dict(a,task=changed))
    old=module.reader_identity();monkeypatch.setattr(module,'reader_identity',lambda:dict(old,new='bad'))
    with pytest.raises(ValueError,match='identity'):module.publish(root,**a)


def test_parallel_and_serial_are_exactly_equal(tmp_path,monkeypatch):
    a,_,_,_=setup(tmp_path,monkeypatch);first=module.publish(tmp_path/'serial',**a)
    parallel=dict(EXEC,cpu_workers=2)
    second=module.publish(tmp_path/'parallel',**dict(a,execution=parallel))
    assert first['manifest_sha256']==second['manifest_sha256']
    assert second['build_timings']['classify']['parallel'] is True
    assert second['build_timings']['classify']['width']==2
    assert second['build_timings']['classify']['peak_in_flight']<=2


def test_all_hard_all_easy_and_class_floor():
    p={r['id']:[.8,.8] for r in rows()}
    ids,_,summary=sample(rows(),['a','b'],p,OPTS)
    assert len(ids)==20 and summary['selected_easy']==0
    p={r['id']:[.9,.1] if r['intent']=='a' else [.1,.9] for r in rows()}
    ids,_,summary=sample(rows(),['a','b'],p,OPTS)
    assert len(ids)==2 and summary['no_difficult_samples'] and summary['actual_easy_fraction']==1
    assert all(c['selected_easy']==1 for c in summary['by_class'].values())


def test_failed_publication_never_becomes_consumable(tmp_path,monkeypatch):
    a,_,_,_=setup(tmp_path,monkeypatch);root=tmp_path/'view'
    original=module.commit_json
    def fail(path,value):
        if Path(path).name=='consumer.json':raise OSError('disk failure')
        original(path,value)
    monkeypatch.setattr(module,'commit_json',fail)
    with pytest.raises(OSError):module.publish(root,**a)
    assert not root.exists()
    with pytest.raises(ValueError,match='interrupted'):module.publish(root,**a)


def test_cli_prepare_audit_bind_idempotent_and_macros(tmp_path,monkeypatch):
    import tools.reports.prepare_stage2_dataset as cli
    from multimodal_jev.local_config import load_config
    a,_,_,ref=setup(tmp_path,monkeypatch)
    monkeypatch.setattr(cli,'inputs',lambda c,p:(a['rows'],a['task'],a['reference'],a['endpoint']))
    c=dict(output=str(tmp_path/'view'),sampling=OPTS,batch_config=BATCH_CONFIG,
        execution=EXEC,evaluation={'interval_updates':64},receipts={k:str(tmp_path/(k+'-receipt.json')) for k in ('prepare','audit','bind')})
    first=cli.run(c,{})
    assert cli.run(c,{})['reused']
    c['manifest_sha256']=first['manifest_sha256']
    assert cli.run(c,{},'audit')['recordings']==5
    config=dict(stage='task',source_policy_id=ref['identity']['policy_id'],release_sha256=ref['identity']['release_sha256'],reference_dataset=ref)
    path=tmp_path/'stage2.json';path.write_text(json.dumps(config))
    c.update(stage2_config=dict(path=str(path),sha256=file_hash(path)),stage2_output=str(tmp_path/'bound.json'))
    bound=cli.run(c,{},'bind');assert cli.run(c,{},'bind')['bound_config_sha256']==bound['bound_config_sha256']
    assert json.loads(Path(c['stage2_output']).read_text())['stop_update']==2
    template=Path(__file__).parents[1]/'configs/portable/operations/stage2-dataset.json'
    macro=dict(AUDITED_STAGE1_REFERENCE_JOB={},STAGE1_DATASET_ENDPOINT=a['endpoint'],STAGE2_DATASET_OUTPUT=c['output'],
        STAGE2_DATASET_SHA256=c['manifest_sha256'],STAGE2_DATASET_RECEIPTS=c['receipts'],STAGE2_INPUT_CONFIG=c['stage2_config'],STAGE2_BOUND_CONFIG=c['stage2_output'])
    local=tmp_path/'macros.json';local.write_text(json.dumps(dict(schema='jev-local-macros-v1',macros=macro)))
    resolved,_=load_config(template,local)
    assert 'source_queries' not in resolved['batch_config'] and resolved['output']==c['output']


def test_parallel_reference_authentication_matches_serial(tmp_path,monkeypatch):
    from multimodal_jev.context_reference_dataset import ReferenceDataset
    a,_,_,ref=setup(tmp_path,monkeypatch)
    reader=ReferenceDataset(ref['root'],manifest_sha256=ref['manifest_sha256'],expected_identity=ref['identity'],expected_population=ref['population'],execution=dict(EXEC,cpu_workers=2))
    assert all(reader.read_sample(r['id'])==a['reference'].read_sample(r['id']) for r in a['rows'])


def test_new_margin_endpoint_accepted_by_existing_reference_builder(tmp_path,monkeypatch):
    from tools.reports.build_context_reference_dataset import validate_endpoint
    a,checkpoint,_,_=setup(tmp_path,monkeypatch)
    ref=a['reference'];c=dict(stage1_endpoint=checkpoint['updates'],release_sha256=ref.identity['release_sha256'],
        base_id=ref.identity['base_id'],expected_population=20,prompt_layout=checkpoint['contract']['prompt_layout'])
    bundle=dict(policy_id=ref.policy_id,contract=ref.identity['contract'])
    validate_endpoint(checkpoint,c,bundle)
    with pytest.raises(ValueError,match='full InfoNCE Stage1 endpoint'):
        validate_endpoint(checkpoint,dict(c,prompt_layout=None),bundle)
    checkpoint['contract']['objective']=copy.deepcopy(OBJ)
    checkpoint['contract']['objective']['probability_margin']['margin']=0
    with pytest.raises(ValueError):validate_endpoint(checkpoint,c,bundle)


def test_component_seam_refuses_old_full_dataset_budget(tmp_path,monkeypatch):
    from multimodal_jev.stage2_training_view import training_inputs
    a,_,_,ref=setup(tmp_path,monkeypatch);root=tmp_path/'view';receipt=module.publish(root,**a)
    data=module.TrainingDataset(root,receipt['manifest_sha256'],rows=a['rows'],task=a['task'],reference=a['reference'],execution=EXEC)
    config=data.bind(dict(stage='task',source_policy_id=ref['identity']['policy_id'],release_sha256=ref['identity']['release_sha256'],reference_dataset=ref))
    population,batches,_=training_inputs(config,a['rows'],a['task'],reference=a['reference'],execution=EXEC)
    assert set(population)==set(data.population) and len(batches)==2
    with pytest.raises(ValueError,match='budget'):training_inputs(dict(config,stop_update=634),a['rows'],a['task'],reference=a['reference'],execution=EXEC)
    with pytest.raises(ValueError,match='authenticated'):training_inputs(config,a['rows'],a['task'])


def test_cli_inputs_bind_actual_audited_reference_producer(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import tools.reports.prepare_stage2_dataset as cli
    import tools.reports.prepare_context_reference_job as producer_api
    a,checkpoint,_,ref=setup(tmp_path,monkeypatch)
    root=tmp_path/'collected';root.mkdir()
    package=tmp_path/'package';(package/'initial').mkdir(parents=True)
    commit=package/'initial/endpoint-commit.json';commit.write_text(json.dumps(checkpoint))
    h=file_hash(commit)
    reference_root=Path(ref['root']);m=json.loads((reference_root/'MANIFEST.json').read_text())
    m['producer']=dict(stage1_commit_sha256=h,endpoint=checkpoint['updates'])
    (reference_root/'MANIFEST.json').write_text(json.dumps(m));mh=file_hash(reference_root/'MANIFEST.json')
    (root/'reference-build').mkdir()
    import shutil
    shutil.copytree(reference_root,root/'reference-build/dataset')
    (root/'reference-build/result.json').write_text(json.dumps(dict(identity=ref['identity'],population=ref['population'],manifest_sha256=mh)))
    (package/'reference-build-config.json').write_text(json.dumps(dict(release='/fixture',release_sha256=ref['identity']['release_sha256'],expected_population=20)))
    proof=dict(stage1_commit_sha256=h,policy_id=ref['identity']['policy_id'],manifest_sha256=mh,recordings=20)
    monkeypatch.setattr(producer_api,'audit_results',lambda c,p:{'output_audit':proof})
    class Handle:
        def load(self,*args):return a['rows'],a['task'],{},False
    monkeypatch.setattr(cli,'resolve',lambda *args:Handle)
    c=dict(schema='jev-stage2-dataset-preparation-v2',execution=EXEC,sampling=OPTS,
        reference_job=dict(collection=str(root),output=str(package)),
        stage1_expected=dict(policy_id=ref['identity']['policy_id'],stage1_commit_sha256=h,endpoint=checkpoint['updates']))
    rr,tt,reader,endpoint=cli.inputs(c,{})
    assert len(rr)==20 and endpoint['stage1_commit_sha256']==h
    proof['stage1_commit_sha256']='0'*64
    with pytest.raises(ValueError,match='producer'):cli.inputs(c,{})


def test_one_action_authenticates_once_and_independently_reconstructs(tmp_path,monkeypatch):
    import tools.reports.prepare_stage2_dataset as cli
    a,_,_,ref=setup(tmp_path,monkeypatch);calls=[]
    def inputs(c,p):calls.append('source');return a['rows'],a['task'],a['reference'],a['endpoint']
    original=module.expected_files
    def sample(*args):calls.append('sample');return original(*args)
    monkeypatch.setattr(cli,'inputs',inputs);monkeypatch.setattr(module,'expected_files',sample)
    path=tmp_path/'base';path.write_text(json.dumps(dict(stage='task',source_policy_id=a['reference'].policy_id,
        release_sha256=ref['identity']['release_sha256'],reference_dataset=ref)))
    c=dict(output=str(tmp_path/'view'),stage2_output=str(tmp_path/'bound'),sampling=OPTS,batch_config=BATCH_CONFIG,
        execution=EXEC,evaluation={'interval_updates':64},receipts={'release':str(tmp_path/'receipt')},
        stage2_config=dict(path=str(path),sha256=file_hash(path)))
    proof=cli.run(c,{},'release')
    assert calls==['source','sample','sample'] and not proof['training_dispatch_allowed']
    assert json.loads(Path(c['stage2_output']).read_bytes())['training_dataset']['manifest_sha256']==proof['manifest_sha256']
    calls.clear()
    bad=dict(c,stage2_config=dict(path=str(path),sha256='0'*64))
    with pytest.raises(ValueError,match='changed'):cli.run(bad,{},'release')
    with pytest.raises(ValueError,match='disjoint'):cli.run(dict(c,stage2_output=str(path)),{},'release')
    assert not calls


@pytest.mark.parametrize('fault',['none','old_endpoint','missing_expected','wrong_plan','invalid_hash'])
def test_endpoint_layout_bound_independently_of_same_step_and_policy(tmp_path,monkeypatch,fault):
    a,point,expected,_=setup(tmp_path,monkeypatch)
    layout=dict(design='1'*64,plan='2'*64,cache='3'*64)
    if fault!='old_endpoint':point['contract']['stage1_layout']=dict(layout)
    if fault!='missing_expected':expected['stage1_layout']=dict(layout)
    if fault=='wrong_plan':expected['stage1_layout']['plan']='4'*64
    if fault=='invalid_hash':
        point['contract']['stage1_layout']['cache']=expected['stage1_layout']['cache']='invalid'
    def bind():return module.endpoint_binding(point,commit_sha256='f'*64,reference=a['reference'],
                                              source_expected=expected,sampling=OPTS)
    if fault=='none':assert bind()['stage1_layout']==layout
    else:
        with pytest.raises(ValueError,match='Stage1 endpoint'):bind()


def test_difficulty_mining_accepts_real_source_width_and_rejects_false_width(tmp_path,monkeypatch):
    a,checkpoint,expected,_=setup(tmp_path,monkeypatch)
    cfg=checkpoint['contract']
    cfg['batch_config']['training_devices']=cfg['world_size']=8
    checkpoint['updates']=expected['endpoint']=module.derive(cfg['batch_config'])['total_updates']
    def bind():
        return module.endpoint_binding(checkpoint,commit_sha256='f'*64,reference=a['reference'],source_expected=expected,sampling=OPTS)
    assert bind()['policy_id']==a['reference'].policy_id
    cfg['world_size']=4
    with pytest.raises(ValueError):bind()
