import copy
import json

import pytest
import torch

from multimodal_jev.context_reference_dataset import ReferenceDataset,publish,digest,sha
from multimodal_jev.span_request_identity import tensor_identity


def fixture(tmp_path,monkeypatch):
    import multimodal_jev.context_reference_dataset as m
    import multimodal_jev.local_checkpoint as ckpt
    monkeypatch.setattr(m,'check_paths',lambda _:None)
    monkeypatch.setattr(ckpt,'check_paths',lambda _:None)
    inputs=dict(input_ids=torch.tensor([[1,2,3]]),attention_mask=torch.ones(1,3,dtype=torch.long),
        input_features=torch.ones(1,2,3),input_features_mask=torch.ones(1,3,dtype=torch.long))
    pos=torch.tensor([[1,2]])
    request=dict(policy_id='a'*64,positions=[1,2],inputs={k:tensor_identity(v) for k,v in sorted(inputs.items())})
    record=dict(request_identity=request,policy_id='a'*64,request_id=digest(request),prob=[.25,.75])
    population={'sample0':record['request_id']}
    identity=dict(schema='jev-reference-dataset-v1',policy_id='a'*64,base_id='b'*64,
        contract={'base_id':'b'*64},runtime_identity=dict(image_sha256='1'*64,source_manifest_sha256='2'*64,
        processor_sha256='3'*64,profile='test',output_protocol='test'),release_sha256='c'*64,
        candidate_ids=['a','b'],population_sha256=digest(population))
    root=tmp_path/'dataset'
    result=publish(root,identity=identity,records=[record],population=population,producer={'test':True})
    kwargs=dict(manifest_sha256=result['manifest_sha256'],expected_identity=identity,expected_population=population)
    return root,kwargs,inputs,pos,record


def test_readonly_exact_results_independent_of_epochs_and_no_inference(tmp_path,monkeypatch):
    root,kw,x,pos,record=fixture(tmp_path,monkeypatch)
    reader=ReferenceDataset(root,**kw)
    first=reader.read(x,pos);assert first==record
    first['prob'][0]=.5
    assert reader.read(x,pos)==ReferenceDataset(root,**kw).read(x,pos)==record
    with pytest.raises(ValueError,match='read-only'):reader.write(record,x,pos)
    x['input_features'][0,0,0]=2
    with pytest.raises(ValueError,match='miss'):reader.read(x,pos)


@pytest.mark.parametrize('fault',['policy','runtime','order','population','content','missing'])
def test_stale_reference_or_corruption_rejected_before_training(tmp_path,monkeypatch,fault):
    root,kw,_,_,_=fixture(tmp_path,monkeypatch);kw=copy.deepcopy(kw)
    if fault=='policy':kw['expected_identity']['policy_id']='d'*64
    if fault=='runtime':kw['expected_identity']['runtime_identity']['profile']='other'
    if fault=='order':kw['expected_identity']['candidate_ids'].reverse()
    if fault=='population':kw['expected_population']['wrong']='e'*64
    if fault=='content':(root/'records.json').write_text('{}')
    if fault=='missing':(root/'population.json').write_text('{}')
    with pytest.raises(ValueError):ReferenceDataset(root,**kw)


def test_complete_coverage_required_even_if_publication_rehashed(tmp_path,monkeypatch):
    root,kw,_,_,_=fixture(tmp_path,monkeypatch)
    (root/'records.json').write_text('{}')
    manifest=json.loads((root/'MANIFEST.json').read_text())
    manifest['files']['records.json']=sha(root/'records.json')
    (root/'MANIFEST.json').write_text(json.dumps(manifest))
    kw['manifest_sha256']=sha(root/'MANIFEST.json')
    with pytest.raises(ValueError,match='incomplete'):ReferenceDataset(root,**kw)


@pytest.mark.parametrize('formal',[False,True])
def test_shared_trainer_binds_cache_in_checkpoint_and_uses_readonly_loader(formal):
    from pathlib import Path
    from tools.reports.context_cached_reference_trainer import generate
    source=generate(Path(__file__).resolve().parents[1],formal=formal)
    assert 'cache=RuntimeReferenceCache(' not in source
    assert 'cache=load_reference(' in source and "contract['reference_dataset']" in source
    assert "set(c['reference_dataset']['population'])!=set(population)" in source
    assert 'ReadOnlyEncodedAudioCache as EncodedAudioCache' in source
    assert "require_training_storage(c['encoded_audio_cache'])" not in source
    assert 'factory=no_online_processor' in source
    assert "contract['stage2_objective']=objective_contract" in source
    compile(source,'<test>','exec')


def objective_config():
    from pathlib import Path
    from tools.reports.context_cached_reference_trainer import BASELINE_OBJECTIVE
    stage=json.loads((Path(__file__).resolve().parents[1]/
        'configs/experiments/context-two-stage-v0.0.11.json').read_text())['stage2']
    return dict(stage2_objective={key:stage[key] for key in BASELINE_OBJECTIVE},
        algorithm=stage['algorithm'],dual_arm=stage['arm'])


def test_stage2_objective_matches_current_design_without_silent_tuning():
    from tools.reports.context_cached_reference_trainer import validate_objective
    c=objective_config()
    checked=validate_objective(c)
    assert checked==c['stage2_objective'] and checked is not c['stage2_objective']
    for key,value in c['stage2_objective'].items():
        bad=copy.deepcopy(c)
        bad['stage2_objective'][key]=value+1 if isinstance(value,(int,float)) else 'other'
        with pytest.raises(ValueError,match='unsupported'):validate_objective(bad)
    for override in ({'learning_rate':.01},{'dual_arm':'ranking'},
                     {'stage2_objective':{}},{'group_size':True}):
        with pytest.raises(ValueError):validate_objective(dict(c,**override))


@pytest.mark.parametrize('formal',[False,True])
def test_invalid_objective_rejected_before_npu_imports_or_output(formal,tmp_path):
    from pathlib import Path
    from tools.reports.context_cached_reference_trainer import generate
    namespace={'__name__':'test_cached_trainer'}
    exec(compile(generate(Path(__file__).resolve().parents[1],formal=formal),
        '<test>','exec'),namespace)
    c=objective_config();c['stage2_objective']['clip_ratio']=.9
    with pytest.raises(ValueError,match='unsupported Stage2 objective: clip_ratio'):
        namespace['run'](c,tmp_path/'not-created')
    assert not (tmp_path/'not-created').exists()


def test_prebuilt_inputs_hit_and_miss_never_preprocess_or_write(tmp_path,monkeypatch):
    from test_cached_context_preparation import setup
    from multimodal_jev.readonly_encoded_audio_cache import ReadOnlyEncodedAudioCache
    from tools.reports.context_cached_reference_trainer import no_online_processor
    request,audio_hash,factory,calls,_=setup(tmp_path,monkeypatch)
    preparer=factory();original=preparer((request,audio_hash))
    before={p.relative_to(tmp_path):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    preparer.cache=ReadOnlyEncodedAudioCache(preparer.cache.root)
    preparer.processor_factory=no_online_processor
    assert preparer((request,audio_hash)).cache_hit
    assert torch.equal(preparer((request,audio_hash)).inputs['input_features'],original.inputs['input_features'])
    preparer.encoding['max_tokens']+=1
    with pytest.raises(ValueError,match='prebuilt input cache miss'):
        preparer((request,audio_hash))
    assert len(calls)==1
    assert before=={p.relative_to(tmp_path):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_actual_grpo_session_uses_only_behavior_and_fails_miss_before_inference(tmp_path,monkeypatch):
    from test_context_dual_session_candidate import setup,query,REF,P0,RUNTIME
    from tools.reports.context_cached_reference_session import TrainingReferenceDataset
    from tools.reports.context_dual_session_candidate import DualContextRolloutSession,probability_record
    from test_context_rollout_session import BASE,CONTRACT,spec
    import multimodal_jev.context_reference_dataset as m
    import multimodal_jev.local_checkpoint as ckpt
    monkeypatch.setattr(m,'check_paths',lambda _:None)
    monkeypatch.setattr(ckpt,'check_paths',lambda _:None)
    _,engine,cache=setup(tmp_path/'online',monkeypatch,policy=REF)
    item=query(P0)
    item['inputs'].update(attention_mask=torch.ones_like(item['inputs']['input_ids']),
        input_features_mask=torch.ones(1,item['inputs']['input_features'].shape[-1],dtype=torch.long))
    record=probability_record(engine.score_many([{k:item[k] for k in ('inputs','positions')}],audio_token_id=1)[0])
    population={'sample0':record['request_id']}
    identity=dict(schema='jev-reference-dataset-v1',policy_id=REF,base_id=BASE,contract=CONTRACT,
        runtime_identity=RUNTIME,release_sha256='c'*64,candidate_ids=['b','a'],population_sha256=digest(population))
    root=tmp_path/'published';receipt=publish(root,identity=identity,records=[record],population=population,producer={})
    cached=TrainingReferenceDataset(root,manifest_sha256=receipt['manifest_sha256'],
        expected_identity=identity,expected_population=population)
    engine.bundle['policy_id']=P0;engine.calls=[];engine.switches=[]
    session=DualContextRolloutSession(engine,cached,runtime_identity=RUNTIME,candidate_ids=['b','a'],
        reference=spec(REF),initial_policy=spec(P0),audio_token_id=1)
    data,_,receipt=session.prepare([item],update=0)
    assert engine.calls==[(P0,1)] and not engine.switches
    assert receipt['reference_misses']==receipt['bulk_reference_calls']==0
    assert torch.equal(data[0]['reference_probability'],torch.tensor(record['prob']))
    from tools.reports.audit_context_cached_training import ReferenceConsumption
    verifier=ReferenceConsumption(dict(manifest_sha256=sha(root/'MANIFEST.json'),
        identity=identity,population=population),root)
    verifier.check(receipt,dict(records=[dict(id='sample0',request_id=item['request_id'])]))
    assert verifier.samples==1
    # Fresh session with a changed actual audio must stop before behavior too.
    session=DualContextRolloutSession(engine,cached,runtime_identity=RUNTIME,candidate_ids=['b','a'],
        reference=spec(REF),initial_policy=spec(P0),audio_token_id=1)
    changed=copy.deepcopy(item);changed['inputs']['input_features'].add_(1)
    with pytest.raises(ValueError,match='miss'):session.prepare([changed],update=0)
    assert engine.calls==[(P0,1)] and session.phase=='failed'
