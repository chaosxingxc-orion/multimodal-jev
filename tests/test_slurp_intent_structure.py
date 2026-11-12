from dataclasses import asdict
import copy
import pytest
from multimodal_jev.task_schema import Candidate,DecisionSchema
from multimodal_jev.thinking import ThinkingSpec
from multimodal_jev.slurp_training import content_hash,project
from multimodal_jev.slurp_intent_structure import compile_intent,training_targets


def fixture():
    pairs={'alarm_query':['alarm','query'],'alarm_set':['alarm','set'],'calendar_query':['calendar','query']}
    task=dict(kind='intent',candidate_pairs=pairs,schema=asdict(DecisionSchema('intent','choice','Choose intent',tuple(Candidate(k,k) for k in pairs),allow_unknown=False)),thinking=ThinkingSpec('v1',('Listen to audio.',)).to_dict())
    req=dict(id='slurp:one',official_split='train',role='learning',source_group='group',audio_sha256='a'*64,input={'audio':'/slurp/audio.flac'})
    target=dict(id=req['id'],request_sha256=content_hash(req),annotation_status='available',native_target={'scenario':'alarm','action':'set'})
    review=dict(status='reviewed',value='alarm_set',request_sha256=content_hash(req),native_target_sha256=content_hash(target),task_sha256=content_hash(task),reviewer='test',evidence=['explicit set alarm'])
    examples,_=project(req,target,{'intent':task},{'intent':review},'b'*64)
    return task,examples[0]


def test_matched_final_and_process_credit_budget():
    task,example=fixture()
    compiled=[training_targets(example,task,arm) for arm in ('D','Q','P')]
    assert all(r.questions[-1]==compiled[0][0].questions[-1] for r,_ in compiled)
    assert compiled[1][0]==compiled[2][0]
    assert compiled[0][1]['span_weights']==[1.]
    assert compiled[1][1]['span_weights']==[0.,0.,1.]
    assert compiled[2][1]['span_weights']==[.1,.1,1.]
    assert compiled[2][1]['target_values']==['alarm','set','alarm_set']
    assert compiled[2][1]['targets']==[0,1,1]
    assert set(compiled[2][0].model_input())=={'audio','questions','final_id','requirements','version'}


def test_model_configuration_does_not_consume_example_or_target():
    task,example=fixture()
    request=compile_intent(example.audio,task,'P')
    assert request==training_targets(example,task,'P')[0]
    altered=copy.deepcopy(task);altered['schema']['question']='Changed'
    with pytest.raises(ValueError,match='match admitted'):training_targets(example,altered,'P')
    for weight in (-1,1,float('nan')):
        with pytest.raises(ValueError):training_targets(example,task,'P',weight)


def test_unreviewed_joint_label_cannot_create_process_labels():
    task,example=fixture();example.metadata['optimization_allowed']=False
    with pytest.raises(ValueError):training_targets(example,task,'P')
