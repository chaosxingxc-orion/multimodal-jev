from dataclasses import replace
import pytest
from multimodal_jev.audio_policy import encode_prompt,vllm_request,system_text
from multimodal_jev.task_schema import Candidate,DecisionSchema,JevExample


class Processor:
    def apply_chat_template(self,messages,**kwargs):
        return messages[0]['content'][0]['text']+'<|audio_start|><|audio_pad|><|audio_end|>'


def example():
    return JevExample('x','train','fixture.wav',DecisionSchema('s','choice','Pick',
                      (Candidate('a','A'),Candidate('b','B'))),'a',{'transcript_reference':'SECRET'})


def test_actor_rollout_prompt_has_schema_but_not_supervision():
    ex=example();audio=object();processor=Processor()
    request=vllm_request(processor,ex,audio)
    assert request['prompt']==encode_prompt(processor,ex,audio)
    assert system_text(ex) in request['prompt']
    assert 'SECRET' not in request['prompt']
    assert request['multi_modal_data']['audio']==(audio,16000)
    assert request['prompt']==vllm_request(processor,replace(ex,target='b'),audio)['prompt']


def test_dropped_schema_is_rejected():
    class BadProcessor:
        def apply_chat_template(self,*args,**kwargs):return '<|audio_pad|>'
    with pytest.raises(ValueError,match='dropped'):encode_prompt(BadProcessor(),example(),object())


def test_unknown_audio_input_mode_rejected():
    with pytest.raises(ValueError,match='input mode'):
        vllm_request(Processor(),example(),object(),'unsupported')
