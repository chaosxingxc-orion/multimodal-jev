import hashlib,json
from dataclasses import asdict
import pytest
from multimodal_jev.consumption import canonical
from multimodal_jev.generative_teacher_feedback import build_feedback,prediction_status


def fixture():
    request=dict(id='g:noul.True',split='train',source_group='g',audio_sha256='sha',
                 input={'audio':'/private/audio.wav','schema':{'schema_id':'s','decision_type':'noul','question':'Is it on?',
                        'candidates':[],'levels':[],'allow_unknown':True}})
    # Use the project's own schema serialization to keep the real contract.
    from multimodal_jev.task_schema import DecisionSchema
    request['input']['schema']=asdict(DecisionSchema('s','noul','Is it on?',allow_unknown=True))
    digest=hashlib.sha256(canonical(request)).hexdigest()
    target=dict(id=request['id'],target=True,request_sha256=digest,metadata={'variant':'noul.True'})
    a=dict(raw_text='```json\n{"value":true}\n```',request_sha256=digest,length_limited=False,
           example_id=request['id'],replicate=0,audio_condition='actual')
    z=dict(raw_text='{"value":null}',request_sha256=digest,length_limited=False,
           example_id=request['id'],replicate=0,audio_condition='zeros')
    return request,target,a,z


def test_feedback_preserves_format_failure_and_has_no_gold_for_silence():
    r,t,a,z=fixture()
    assert prediction_status(r,a,t)['correct'] is False
    assert prediction_status(r,z)['correct'] is None
    kwargs=dict(parent_instruction='listen',parent_prompt_sha256=hashlib.sha256(b'listen').hexdigest(),actor_revision='r',metrics={},sensitivity={})
    context=build_feedback([r],[t],{r['id']:a},{r['id']:z},**kwargs)
    assert context['representatives'][0]['zeros_status']['correct'] is None
    assert context['representatives'][0]['actual_raw']==a['raw_text']
    assert context['zeros_semantic_gold_available'] is False
    assert '/private/audio.wav' not in json.dumps(context)
    with pytest.raises(ValueError,match='TRAIN'):
        build_feedback([{**r,'split':'test'}],[t],{r['id']:a},{r['id']:z},**kwargs)
    with pytest.raises(ValueError,match='custody'):
        build_feedback([r],[{**t,'request_sha256':'changed'}],{r['id']:a},{r['id']:z},**kwargs)
    with pytest.raises(ValueError,match='instruction hash'):
        build_feedback([r],[t],{r['id']:a},{r['id']:z},**{**kwargs,'parent_instruction':'changed'})
    with pytest.raises(ValueError,match='waveform condition'):
        build_feedback([r],[t],{r['id']:a},{r['id']:{**z,'audio_condition':'actual'}},**kwargs)
    with pytest.raises(ValueError,match='invalid task target'):
        build_feedback([r],[{**t,'target':1}],{r['id']:a},{r['id']:z},**kwargs)
