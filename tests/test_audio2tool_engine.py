import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from multimodal_jev.audio2tool_engine import accept_response,validate_request


def fixtures():
    ids=[1,2,3]
    prior=dict(input_tokens=3,input_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
    response=SimpleNamespace(finished=True,prompt_token_ids=ids,
        outputs=[SimpleNamespace(token_ids=[42],finish_reason='length')])
    return prior,response


def test_success_and_early_eos_are_capacity_only():
    p,r=fixtures()
    assert accept_response([r],p)['token_ids']==[42]
    r.outputs[0].finish_reason='stop'
    assert accept_response([r],p)['finish_reason']=='stop'


@pytest.mark.parametrize('mutation',['ids','unfinished','multiple','empty','extra','abort','boolean'])
def test_engine_response_mismatch_rejected(mutation):
    p,r=fixtures()
    if mutation=='ids':r.prompt_token_ids=[1,3,2]
    if mutation=='unfinished':r.finished=False
    if mutation=='multiple':r.outputs*=2
    if mutation=='empty':r.outputs[0].token_ids=[]
    if mutation=='extra':r.outputs[0].token_ids=[1,2]
    if mutation=='abort':r.outputs[0].finish_reason='abort'
    if mutation=='boolean':r.outputs[0].token_ids=[True]
    with pytest.raises(ValueError):accept_response([r],p)


def test_prompt_audio_receipts_and_rates_bound_to_prior():
    prompt='fixed native request';receipts=[dict(actor_waveform_sha256='a',actor_samples=160)]
    p=dict(prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),audio_receipts=receipts)
    request=dict(prompt=prompt,multi_modal_data=dict(audio=[(object(),16000)]))
    validate_request(request,receipts,p)
    for mutation in ('prompt','receipt','rate','count'):
        r=copy.deepcopy(request);a=copy.deepcopy(receipts)
        if mutation=='prompt':r['prompt']+='changed'
        if mutation=='receipt':a[0]['actor_waveform_sha256']='b'
        if mutation=='rate':r['multi_modal_data']['audio']=[(object(),24000)]
        if mutation=='count':r['multi_modal_data']['audio']=[]
        with pytest.raises(ValueError):validate_request(r,a,p)
