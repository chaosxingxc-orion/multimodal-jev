import asyncio
from copy import deepcopy
import hashlib
from types import SimpleNamespace as NS

import pytest

import multimodal_jev.omni_async_rollout as module
from test_omni_rollout_records import setup


def fixture(monkeypatch, mode='good'):
    args,row,decision,prepared,response,kwargs=setup()
    prepared['prompt_sha256']=hashlib.sha256(b'raw audio prompt').hexdigest()
    response.request_id='request-1'
    response.outputs.reverse()
    waves=[object(),object()]
    request=dict(prompt='raw audio prompt',multi_modal_data={'audio':[(w,16000) for w in waves]})
    monkeypatch.setattr(module,'vllm_request',lambda *args:request)
    class Engine:
        def __init__(self):
            self.calls=[];self.aborts=[]
            self.model_config=NS(logprobs_mode='raw_logprobs',generation_config='vllm',max_model_len=4096)
        async def generate(self,**kw):
            self.calls.append(kw)
            if mode=='error':raise RuntimeError('engine failed')
            if mode=='cancel':raise asyncio.CancelledError()
            if mode=='empty':return
            partial=deepcopy(response);partial.finished=False
            yield partial
            final=deepcopy(response)
            if mode=='unfinished':final.finished=False
            if mode=='identity':final.request_id='other'
            if mode=='prompt':final.prompt_token_ids.reverse()
            if mode=='missing':final.outputs.pop()
            if mode=='logprobs':final.outputs[0].logprobs=None
            yield final
        async def abort(self,request_id):self.aborts.append(request_id)
    engine=Engine()
    sampler=module.OmniAsyncSampler(engine,NS(tokenizer=args['tokenizer']),
        policy_sha256=kwargs['policy_sha256'],instruction_sha256=kwargs['instruction_sha256'],
        sampling_params_factory=lambda **kw:NS(**kw))
    call=lambda: sampler.sample(row,decision,prepared,waves,request_id='request-1')
    return engine,sampler,prepared,request,call


def test_complete_k_preserves_audio_raw_prompt_and_zero_logprobs(monkeypatch):
    engine,sampler,prepared,request,call=fixture(monkeypatch)
    records=asyncio.run(call())
    assert [r['replicate'] for r in records]==[0,1]
    assert all(r['log_probs'] for r in records)
    assert engine.calls[0]['prompt'] is request
    assert len(engine.calls[0]['prompt']['multi_modal_data']['audio'])==2
    assert engine.calls[0]['sampling_params'].logprobs==0
    assert engine.calls[0]['sampling_params'].n==2 and not engine.aborts
    with pytest.raises(ValueError,match='no replay'):asyncio.run(call())
    assert len(engine.calls)==1


@pytest.mark.parametrize('mode',['error','cancel','empty','unfinished','identity','prompt','missing','logprobs'])
def test_no_partial_group_or_ambiguous_retry(monkeypatch,mode):
    engine,sampler,prepared,request,call=fixture(monkeypatch,mode)
    with pytest.raises((ValueError,RuntimeError,asyncio.CancelledError)):asyncio.run(call())
    assert engine.aborts==(['request-1'] if mode in ('error','cancel','empty','unfinished','identity') else [])
    with pytest.raises(ValueError,match='no replay'):asyncio.run(call())
    assert len(engine.calls)==1


@pytest.mark.parametrize('change',['context','prompt','ids','row','probability_mode'])
def test_admission_fails_before_dispatch(monkeypatch,change):
    engine,sampler,prepared,request,call=fixture(monkeypatch)
    if change=='context':sampler.max_model_len=4
    if change=='prompt':request['prompt']='changed'
    if change=='ids':prepared['input_ids'][0]=10
    if change=='row':prepared['request_sha256']='0'*64
    if change=='probability_mode':engine.model_config.logprobs_mode='processed_logprobs'
    with pytest.raises(ValueError):asyncio.run(call())
    assert not engine.calls and not sampler.dispatched
