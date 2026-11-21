from dataclasses import replace
import pytest
import torch
from test_context_decision_contract import request,Processor
from multimodal_jev.context_decision_contract import PREFIX_LAYOUT,DECISION_TOKEN,render_prefix_context_request,encode_context_request
from multimodal_jev.cached_context_preparation import CachedContextPreparer


def test_static_prefix_shared_across_audio_and_history():
 a=request();b=replace(a,audio='/different.wav',history=('Different history.',))
 pa,ha,sa=render_prefix_context_request(a);pb,hb,sb=render_prefix_context_request(b)
 assert pa==pb and sa==sb and ha!=hb
 assert a.audio not in pa and b.audio not in pb and 'Earlier supplied' not in pa
 assert all(q.definition in pa and q.question in pa for q in a.questions)
 assert DECISION_TOKEN not in pa and sa.count(DECISION_TOKEN)==2


def test_changed_task_changes_prefix():
 a=request();b=replace(a,questions=(replace(a.questions[0],definition='Changed rule'),a.questions[1]))
 assert render_prefix_context_request(a)[0]!=render_prefix_context_request(b)[0]


def test_encoded_markers_after_audio_and_rules_before_audio():
 p=Processor();r=replace(request(),history=())
 batch,pos=encode_context_request(p,r,[0.],decision_token_id=998,audio_token_id=999,max_tokens=4096,prompt_layout=PREFIX_LAYOUT)
 ids=batch['input_ids'][0];audio=(ids==999).nonzero().flatten()
 prefix=p.tokenizer.encode(render_prefix_context_request(r)[0]);assert ids[:len(prefix)].tolist()==prefix
 assert int(audio[-1])<int(pos[0,0]) and len(pos[0])==2


def test_layout_part_of_preprocessing_cache_identity_and_old_default_unchanged():
 kwargs=dict(processor_sha256='a'*64,decision_token_id=998,audio_token_id=999,max_tokens=4096)
 old=CachedContextPreparer(None,None,**kwargs);new=CachedContextPreparer(None,None,**kwargs,prompt_layout=PREFIX_LAYOUT)
 a=old.identity(request(),'b'*64);b=new.identity(request(),'b'*64)
 assert 'prompt_layout' not in a['context_encoding']
 assert b['context_encoding']['prompt_layout']==PREFIX_LAYOUT and a!=b
 with pytest.raises(ValueError):CachedContextPreparer(None,None,**kwargs,prompt_layout='unknown')
