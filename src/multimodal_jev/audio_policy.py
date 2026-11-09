"""One prompt/answer contract for HF actor inputs and vLLM audio requests.

This is a component for a future verl agent loop; it is not a full verl worker.
Never pass expanded HF audio token IDs back to a processor that expands them.
"""
from __future__ import annotations
import json
from .task_schema import render_schema


INSTRUCTION = ('Use the audio evidence and the decision definition below. '
               'Return exactly one JSON object with key "value". '
               'Use a candidate id for choice, a boolean for noul, '
               'an allowed numeric level for score, or a string for open_slot. '
               'Use null only when unknown is allowed and evidence or a valid candidate is missing.\n')


def system_text(example):
    return 'English\n' + INSTRUCTION + render_schema(example.schema)


def messages(example, audio):
    return [{'role':'system','content':[{'type':'text','text':system_text(example)}]},
            {'role':'user','content':[{'type':'audio','audio':audio}]}]


def encode_prompt(processor, example, audio):
    text=processor.apply_chat_template(messages(example,audio),tokenize=False,add_generation_prompt=True)
    if system_text(example) not in text:
        raise ValueError('processor dropped schema from prompt')
    if text.count('<|audio_pad|>') != 1:
        raise ValueError('single audio placeholder required before expansion')
    return text


def load_audio(path):
    import numpy as np
    import soundfile as sf
    audio,rate=sf.read(path,dtype='float32')
    if rate!=16000 or audio.ndim!=1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError('audio must be finite nonempty mono 16 kHz')
    return audio


def hf_prompt(processor, example, audio):
    text=encode_prompt(processor,example,audio)
    return processor(text=[text],audio=[audio],sampling_rate=16000,return_tensors='pt')


def vllm_request(processor, example, audio):
    # vLLM owns audio token expansion. Raw prompt text is intentional.
    return {'prompt':encode_prompt(processor,example,audio),
            'multi_modal_data':{'audio':(audio,16000)}}


def answer_ids(tokenizer, value):
    answer=json.dumps({'value':value},ensure_ascii=False,separators=(',',':'))
    tokens=tokenizer.encode(answer,add_special_tokens=False)
    end=tokenizer.convert_tokens_to_ids('<|im_end|>')
    if not isinstance(end,int) or end==tokenizer.unk_token_id:
        raise ValueError('Qwen assistant end token required')
    return tokens+[end]


def teacher_forcing(processor, example, audio):
    import torch
    inputs=dict(hf_prompt(processor,example,audio))
    prompt_length=inputs['input_ids'].shape[-1]
    response=torch.tensor([answer_ids(processor.tokenizer,example.target)],dtype=torch.long)
    inputs['input_ids']=torch.cat((inputs['input_ids'],response),dim=-1)
    inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
    inputs['position_ids']=torch.arange(inputs['input_ids'].shape[-1]).unsqueeze(0)
    inputs['labels']=inputs['input_ids'].clone()
    inputs['labels'][:,:prompt_length]=-100
    # Processor modality masks are prompt-only and must never silently retain
    # their old lengths when the response is appended.
    for key in set(inputs)-{'input_ids','attention_mask','position_ids','labels',
                            'input_features','input_features_mask'}:
        raise ValueError(f'unhandled processor output: {key}')
    return inputs,prompt_length


def response_log_probs(logits, input_ids, prompt_length):
    """Causal log-probs including the assistant end token, without prompt loss."""
    import torch
    if prompt_length<1 or prompt_length>=input_ids.shape[-1]:
        raise ValueError('invalid response boundary')
    prediction=logits[:,prompt_length-1:-1].float()
    target=input_ids[:,prompt_length:]
    return torch.log_softmax(prediction,dim=-1).gather(-1,target.unsqueeze(-1)).squeeze(-1)
