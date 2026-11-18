"""Lossless attribute-request representation; independent of frozen renderer."""
import json
from multimodal_jev.context_decision_contract import (
    ApplicabilityQuestion, ContextDecisionRequest, DECISION_TOKEN, RESERVED_TIMESTAMP_TOKEN,
    render_context_request)

CATALOG_PREFIX='Shared complete benchmark event catalog: '
GROUP_PREFIX='This group consists exactly of event IDs: '


def pack(request):
    requirements=list(request.requirements)
    if not requirements[-1].startswith(CATALOG_PREFIX):raise ValueError('Explicit shared catalog required')
    events=json.loads(requirements.pop()[len(CATALOG_PREFIX):])
    if any(set(e)!={'id','description'} for e in events):raise ValueError('Known event schema required')
    ids=[e['id'] for e in events]
    if len(set(ids))!=len(ids):raise ValueError('Unique event IDs required')
    questions={q.question for q in request.questions}
    if len(questions)!=1:raise ValueError('One shared question required')
    groups=[]
    for q in request.questions:
        if not q.definition.startswith(GROUP_PREFIX) or not q.definition.endswith('.'):
            raise ValueError('Explicit group definition required')
        members=json.loads(q.definition[len(GROUP_PREFIX):-1])
        if not members or len(set(members))!=len(members) or any(m not in ids for m in members):
            raise ValueError('Complete known membership required')
        groups.append([q.id,[ids.index(m) for m in members]])
    value=dict(version=request.version,requirements=requirements,history=list(request.history),
               events=[[e['id'],e['description']] for e in events],question=next(iter(questions)),groups=groups)
    if unpack(request.audio,value).model_input()!=request.model_input():raise ValueError('Lossy request packing')
    return value


def unpack(audio,value):
    if set(value)!={'version','requirements','history','events','question','groups'}:
        raise ValueError('Exact compact schema required')
    events=[dict(id=e[0],description=e[1]) for e in value['events']]
    ids=[e['id'] for e in events]
    if len(set(ids))!=len(ids):raise ValueError('Unique events required')
    questions=[]
    for uid,indices in value['groups']:
        if not indices or len(set(indices))!=len(indices) or any(type(i)!=int or not 0<=i<len(ids) for i in indices):
            raise ValueError('Valid explicit event indices required')
        members=[ids[i] for i in indices]
        questions.append(ApplicabilityQuestion(uid,GROUP_PREFIX+json.dumps(members,separators=(',',':'))+'.',value['question']))
    requirements=(*value['requirements'],CATALOG_PREFIX+json.dumps(events,ensure_ascii=False,separators=(',',':')))
    return ContextDecisionRequest(audio,tuple(questions),requirements,tuple(value['history']),value['version'])


def render(request):
    value=pack(request)
    system,_=render_context_request(request)
    legend=('events rows are [event_id, exact definition]. groups rows are [decision_id, member_event_indices]. '
            'Indices are zero-based positions in events; each group consists exactly of those event IDs. '
            'Apply the shared question to every group. Readout i corresponds to groups[i]. '
            'Return one independent probability per readout; do not normalize across groups.\n')
    body=json.dumps(value,ensure_ascii=False,separators=(',',':'))
    readouts='\n'.join(str(i)+'\n'+DECISION_TOKEN for i in range(len(request.questions)))
    return system,legend+body+'\n'+readouts


def encode(processor,request,audio,*,decision_token_id,audio_token_id,max_tokens):
    import torch
    if (type(max_tokens)!=int or max_tokens<1 or type(decision_token_id)!=int or type(audio_token_id)!=int
            or min(decision_token_id,audio_token_id)<0 or decision_token_id==audio_token_id):
        raise ValueError('Explicit token identities and context bound required')
    tokenizer=processor.tokenizer
    if (tokenizer.get_vocab().get(DECISION_TOKEN)!=decision_token_id or DECISION_TOKEN not in tokenizer.all_special_tokens
            or tokenizer.encode(DECISION_TOKEN,add_special_tokens=False)!=[decision_token_id]):
        raise ValueError('Registered single marker required')
    system,content=render(request)
    messages=[{'role':'system','content':[{'type':'text','text':system}]},
              {'role':'user','content':[{'type':'audio','audio':audio}]},
              {'role':'assistant','content':content}]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
    if system not in text or content not in text or text.count(DECISION_TOKEN)!=len(request.questions):
        raise ValueError('Template changed request')
    batch=dict(processor(text=[text],audio=[audio],sampling_rate=16000,return_tensors='pt'))
    ids=batch['input_ids'];mask=batch['attention_mask']
    if ids.dtype!=torch.long or ids.ndim!=2 or ids.shape[0]!=1 or ids.shape[1]>max_tokens:
        raise ValueError('One complete bounded request required; truncation forbidden')
    reserved=tokenizer.get_vocab().get(RESERVED_TIMESTAMP_TOKEN)
    if reserved is not None and (ids==reserved).any():raise ValueError('Reserved timestamp token')
    positions=(ids[0]==decision_token_id).nonzero().flatten()
    audio_positions=(ids[0]==audio_token_id).nonzero().flatten()
    if len(positions)!=len(request.questions) or not len(audio_positions) or int(audio_positions.max())>=int(positions[0]):
        raise ValueError('Complete ordered marker coverage after audio required')
    if mask.shape!=ids.shape or not ((mask==0)|(mask==1)).all() or not mask[0,positions].all():
        raise ValueError('Valid readout mask required')
    return batch,positions[None,:]
