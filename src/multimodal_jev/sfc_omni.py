"""SFC supplied-history messages for Omni, following the pinned official client.

Only input fields are accepted. Literal <audio> markers in official text remain
unchanged, as in eval/infer.py; actual audio content is attached separately.
This is an input mapping, not evidence of model inference or tool execution.
"""
from .sfc_audio import _text


def messages(inputs, audios):
    if set(inputs)!={'messages','audios'}:
        raise ValueError('only input messages and audios are allowed')
    turns=inputs['messages'];paths=inputs['audios']
    if (not isinstance(turns,list) or len(turns) not in (1,3)
            or not isinstance(paths,list) or len(paths)!=(len(turns)+1)//2
            or len(audios)!=len(paths)):
        raise ValueError('one or two ordered audio turns required')
    for i,turn in enumerate(turns):
        if set(turn)!={'role','content'} or turn['role']!=('user' if i%2==0 else 'assistant'):
            raise ValueError('alternating user/assistant input ending in user required')
        text=_text(turn['content'])
        if text.count('<audio>')!=(1 if i%2==0 else 0):
            raise ValueError('exactly one placeholder in each user turn required')
    result=[{'role':'system','content':turns[0]['content']}]
    for i,wave in enumerate(audios):
        content=[]
        if i:content.append({'type':'text','text':turns[2*i]['content']})
        content.append({'type':'audio','audio':wave})
        result.append({'role':'user','content':content})
        if 2*i+1<len(turns)-1:
            result.append({'role':'assistant','content':turns[2*i+1]['content']})
    return result


def vllm_request(processor, inputs, audios):
    prompt=processor.apply_chat_template(messages(inputs,audios),tokenize=False,add_generation_prompt=True)
    return {'prompt':prompt,'multi_modal_data':{'audio':[(wave,16000) for wave in audios]}}
