from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from multimodal_jev.slurp_training import content_hash, project, validate_admission
from multimodal_jev.task_schema import Candidate, DecisionSchema
from multimodal_jev.thinking import ThinkingSpec


def fixture():
    request = dict(id='slurp:a.flac', official_split='train', role='learning',
                   source_group='g', audio_sha256='a'*64, input={'audio':'/audio/a.flac'})
    target = dict(id=request['id'], request_sha256=content_hash(request),
                  annotation_status='available', optimization_allowed=False,
                  semantic_review_status='pending', native_target=dict(
                      scenario='email', action='query', entities=[{'type':'person','filler':'amy'}]))
    thinking = ThinkingSpec('fixture-v1', ('Check the semantic role in context.',)).to_dict()
    tasks = {
        'intent':dict(kind='intent', thinking=thinking,
            candidate_pairs={'a':['email','query'], 'b':['calendar','query']},
            schema=asdict(DecisionSchema('intent','choice','Choose the requested action.',
                (Candidate('a','Query email'),Candidate('b','Query calendar')),allow_unknown=False))),
        'person':dict(kind='type_presence', entity_type='person', thinking=thinking,
            schema=asdict(DecisionSchema('role','noul','Is a person referenced?',allow_unknown=False))),
        'date':dict(kind='type_presence', entity_type='date', thinking=thinking,
            schema=asdict(DecisionSchema('role','noul','Is a calendar date referenced?',allow_unknown=False))),
    }
    reviews = {k:dict(status='reviewed', request_sha256=content_hash(request),
        native_target_sha256=content_hash(target), task_sha256=content_hash(task),
        value={'intent':'a','person':True,'date':False}[k], reviewer='synthetic-test-fixture',
        evidence=['fixture-only; not actual dataset approval']) for k,task in tasks.items()}
    return request, target, tasks, reviews


def test_partial_review_mask_never_becomes_null_or_negative():
    request,target,tasks,reviews=fixture()
    reviews['date']['status']='unavailable'
    del reviews['person']
    examples,masks=project(request,target,tasks,reviews,'b'*64)
    assert masks=={'intent':'reviewed','person':'pending','date':'unavailable'}
    assert len(examples)==1 and examples[0].target=='a'
    assert target['optimization_allowed'] is False
    assert 'evidence' not in json.dumps(examples[0].model_input())
    assert 'amy' not in json.dumps(examples[0].to_record())
    assert 'slurp_review' not in examples[0].model_input()


def test_explicit_negative_requires_its_own_review():
    request,target,tasks,reviews=fixture()
    examples,masks=project(request,target,tasks,reviews,'b'*64)
    assert [e.target for e in examples]==['a',True,False]
    assert all(validate_admission(e) is None for e in examples)
    reviews['date']['evidence']=[]
    with pytest.raises(ValueError,match='evidence'):
        project(request,target,tasks,reviews,'b'*64)
    assert project(request,target,tasks,{},'b'*64)[0]==[]


@pytest.mark.parametrize('change', ['target','definition','thinking','source','role','conflict','value','leak'])
def test_changed_or_unusable_supervision_rejected(change):
    request,target,tasks,reviews=fixture()
    if change=='target':target['native_target']['entities']=[]
    elif change=='definition':tasks['person']['schema']['question']='Changed definition'
    elif change=='thinking':tasks['person']['thinking']['requirements']=['Different requirement']
    elif change=='source':request['audio_sha256']='c'*64
    elif change=='role':request['role']='selection'
    elif change=='conflict':target['annotation_status']='conflict'
    elif change=='value':reviews['person']['value']=1
    else:request['input']['transcript']='gold text'
    with pytest.raises(ValueError):project(request,target,tasks,reviews,'b'*64)


@pytest.mark.parametrize('change',['target','question','thinking','group','audio_hash','path','permission','review'])
def test_admitted_decision_binding_detects_stale_changes(change):
    request,target,tasks,reviews=fixture()
    ex=project(request,target,tasks,reviews,'b'*64)[0][1]
    if change=='target':ex=replace(ex,target=False)
    elif change=='question':ex=replace(ex,schema=replace(ex.schema,question='Different?'))
    elif change=='thinking':ex=replace(ex,thinking=ThinkingSpec('v2',('Different check.',)))
    elif change=='group':ex.metadata['source_group']='different'
    elif change=='audio_hash':ex.metadata['audio_sha256']='c'*64
    elif change=='path':ex=replace(ex,audio='/other.wav')
    elif change=='permission':ex.metadata['optimization_allowed']=False
    else:ex.metadata.pop('slurp_review')
    with pytest.raises(ValueError):validate_admission(ex)


@pytest.mark.parametrize('entry',['sft','rl'])
@pytest.mark.parametrize('mark',['legacy_slurp','denied','selection'])
def test_training_entrypoints_reject_unapproved_rows(tmp_path,entry,mark):
    from test_rl_preflight import fixture as rl_fixture
    from multimodal_jev.sft import training_batches
    from multimodal_jev.rl_preflight import validate_schedule
    config=rl_fixture(tmp_path)
    path=Path(config['manifest']);row=json.loads(path.read_text())
    if mark=='legacy_slurp':row['schema']['id']='slurp.scenario.v1'
    elif mark=='denied':row['metadata']['optimization_allowed']=False
    else:row['metadata']['role']='selection'
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='SLURP|disallowed|learning'):
        if entry=='sft':training_batches({'train_manifest':str(path),'batches':[['ex']]})
        else:validate_schedule(config)


def test_approved_decisions_feed_explicit_sft_batches(tmp_path):
    from multimodal_jev.sft import training_batches
    request,target,tasks,reviews=fixture()
    examples,_=project(request,target,tasks,reviews,'b'*64)
    path=tmp_path/'train.jsonl'
    path.write_text(''.join(json.dumps(e.to_record())+'\n' for e in examples))
    cfg={'train_manifest':str(path),'batches':[[e.example_id for e in examples]]}
    batch=training_batches(cfg)[0]
    assert [e.target for e in batch]==['a',True,False]


def test_direct_actor_update_rejects_before_runtime_imports():
    from multimodal_jev.rl_update import update_actor
    request,target,tasks,reviews=fixture()
    ex=project(request,target,tasks,reviews,'b'*64)[0][0]
    ex.metadata['optimization_allowed']=False
    with pytest.raises(ValueError,match='disallowed'):
        update_actor(None,None,None,[ex],[],[],'policy',2)
