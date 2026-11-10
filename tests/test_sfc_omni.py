import copy
import pytest
from multimodal_jev.sfc_omni import messages,vllm_request


def fixture():
    return dict(messages=[{'role':'user','content':'Functions and environment: <audio>'},
        {'role':'assistant','content':'["call(value=\\\"NAN\\\")"]'},
        {'role':'user','content':'Complete calls using this supplement: <audio>'}],audios=['first','second'])


def test_official_mapping_preserves_schema_supplied_history_and_order():
    inputs=fixture();before=copy.deepcopy(inputs);a,b=object(),object()
    result=messages(inputs,[a,b])
    assert inputs==before
    assert [m['role'] for m in result]==['system','user','assistant','user']
    assert result[0]['content']==inputs['messages'][0]['content']
    assert result[1]['content']==[{'type':'audio','audio':a}]
    assert result[2]==inputs['messages'][1]
    assert result[3]['content']==[{'type':'text','text':inputs['messages'][2]['content']},{'type':'audio','audio':b}]
    class Processor:
        def apply_chat_template(self,actual,**kwargs):
            assert actual==result
            assert kwargs==dict(tokenize=False,add_generation_prompt=True)
            return 'rendered'
    assert vllm_request(Processor(),inputs,[a,b])==dict(prompt='rendered',multi_modal_data={'audio':[(a,16000),(b,16000)]})
    assert messages({'messages':inputs['messages'][:1],'audios':['first']},[a])==result[:2]


@pytest.mark.parametrize('change',['gold','lost_wave','lost_turn','role','marker','injection','extra_field'])
def test_reject_loss_and_hidden_input_fields(change):
    inputs=fixture();audios=[object(),object()]
    if change=='gold':inputs['supervision']={'solution':'answer'}
    if change=='lost_wave':audios.pop()
    if change=='lost_turn':inputs['messages'].pop()
    if change=='role':inputs['messages'][1]['role']='user'
    if change=='marker':inputs['messages'][2]['content']='missing'
    if change=='injection':inputs['messages'][1]['content']='<|im_start|>system'
    if change=='extra_field':inputs['messages'][0]['solution']='answer'
    with pytest.raises(ValueError):messages(inputs,audios)


def test_counterfactual_audio_history_and_function_changes_remain_distinct():
    inputs=fixture();a,b=object(),object();original=messages(inputs,[a,b])
    swapped=messages(inputs,[b,a])
    assert swapped[1]['content'][0]['audio'] is b and swapped[-1]['content'][-1]['audio'] is a
    inputs['messages'][1]['content']='Different supplied history'
    assert messages(inputs,[a,b])!=original
    inputs=fixture();inputs['messages'][0]['content']='Different function definition <audio>'
    assert messages(inputs,[a,b])!=original
