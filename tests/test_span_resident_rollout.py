"""Real tensor transport with a fake engine; no NPU execution claim."""
import hashlib
import json
from types import SimpleNamespace as NS
import pytest
import torch
import vllm
import vllm.lora.request
import multimodal_jev.span_resident_rollout as module


def fixture(monkeypatch):
    monkeypatch.setattr(module,'checkpoint_identity',lambda p:(p,{}))
    monkeypatch.setattr(module,'adapter_layout',lambda p:['target'])
    monkeypatch.setattr(vllm,'PoolingParams',lambda **kw:NS(**kw))
    monkeypatch.setattr(vllm.lora.request,'LoRARequest',lambda name,slot,path:NS(slot=slot,path=path))
    s=module.ResidentSpanRollout.__new__(module.ResidentSpanRollout)
    s.config=dict(model='base',base_id='base',runtime_profile='profile',max_model_len=4096)
    s.phase='ready';s.targets=['target'];s.slots={};calls=[];active=[None];closed=[]
    def logits(adapter):return torch.tensor([[0.,1. if adapter=='A' else 2.]])
    def encode(prompts,**kw):
        lora=kw['lora_request'];active[0]=lora.slot;calls.append((lora.slot,prompts,kw))
        return [NS(prompt_token_ids=kw['pooling_params'].extra_kwargs['expected_input_ids'],outputs=NS(data=logits(lora.path)))]
    worker=lambda rpc:[dict(registered_adapter_ids=[active[0]],lora_dtype='torch.bfloat16',
        head_projection_dtype='torch.bfloat16',actual_lora_buffer_dtypes={'layer':['torch.bfloat16']})]
    s.engine=NS(encode=encode,llm_engine=NS(collective_rpc=worker,engine_core=NS(shutdown=lambda **kw:closed.append(kw))))
    def inputs(adapter='A'):
        cfg=dict(s.config,adapter=adapter,adapter_id=adapter,population=dict(requests=1))
        row=dict(id='audio');batch=dict(input_ids=torch.tensor([[1,9,9,2]]),input_features=torch.ones(1,3,2),input_features_mask=torch.ones(1,2,dtype=torch.long))
        identity=dict(policy_id='base:'+adapter,candidate_ids=[[4,5]],positions=[[3]],
            inputs={k:module.tensor_identity(v) for k,v in sorted(batch.items())},
            sampling=dict(temperature=1.0,policy='independent_finite_categorical'))
        rid=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        compiled=[(row,batch,torch.tensor([[3]]),[[4,5]],rid,identity,NS(tokenizer=NS(convert_tokens_to_ids=lambda t:9)))]
        ref=dict(passed=True,base_id='base',adapter_id=adapter,records=[dict(id='audio',request_id=rid,request_identity=identity,candidate_logprobs=torch.log_softmax(logits(adapter),-1).tolist())])
        return cfg,compiled,ref
    return s,inputs,calls,closed


def test_new_policies_and_return_reuse_engine_without_stale_slots(monkeypatch):
    s,inputs,calls,closed=fixture(monkeypatch);engine=s.engine
    reports=[s.evaluate(*inputs(a)) for a in ['A','B','A']]
    assert s.engine is engine and [r['slot'] for r in reports]==[1,2,1]
    assert reports[0]['records']==reports[2]['records']
    assert reports[0]['records'][0]['candidate_logprobs']!=reports[1]['records'][0]['candidate_logprobs']
    assert [c[1][0]['prompt_token_ids'] for c in calls]==[[1,9,2]]*3
    assert calls[1][1][0]['multi_modal_uuids']=={'audio':[inputs('B')[2]['records'][0]['request_id']]}
    s.close();s.close();assert closed==[dict(timeout=30.)]
    with pytest.raises(ValueError,match='ready'):s.evaluate(*inputs())


@pytest.mark.parametrize('bad',['request','reference_policy','duplicate','missing','stale_output','worker_slot','worker_dtype','layout','adapter_changed','expanded_ids','features','candidates'])
def test_failure_poisoning_prevents_retry(monkeypatch,bad):
    s,inputs,calls,_=fixture(monkeypatch);cfg,compiled,ref=inputs()
    if bad=='request':ref['records'][0]['request_id']='wrong'
    elif bad=='reference_policy':ref['adapter_id']='wrong'
    elif bad=='duplicate':compiled*=2
    elif bad=='missing':compiled=[]
    elif bad=='stale_output':cfg,compiled,ref=inputs('B');ref['records'][0]['candidate_logprobs']=inputs('A')[2]['records'][0]['candidate_logprobs']
    elif bad.startswith('worker_'):
        original=s.engine.llm_engine.collective_rpc
        def worker(rpc):
            result=original(rpc)
            if bad=='worker_slot':result[0]['registered_adapter_ids']=[999]
            else:result[0]['actual_lora_buffer_dtypes']={'layer':['torch.float32']}
            return result
        s.engine.llm_engine.collective_rpc=worker
    elif bad=='layout':monkeypatch.setattr(module,'adapter_layout',lambda p:['changed'])
    elif bad=='adapter_changed':
        count=[0]
        def identity(p):count[0]+=1;return (p if count[0]==1 else 'changed',{})
        monkeypatch.setattr(module,'checkpoint_identity',identity)
    elif bad=='expanded_ids':
        original=s.engine.encode
        def encode(*a,**kw):r=original(*a,**kw);r[0].prompt_token_ids=[1];return r
        s.engine.encode=encode
    elif bad=='features':compiled[0][1]['input_features'][0,0,0]=2.
    elif bad=='candidates':compiled[0][3][0].reverse()
    with pytest.raises(ValueError):s.evaluate(cfg,compiled,ref)
    assert s.phase=='failed'
    before=len(calls)
    with pytest.raises(ValueError,match='no retry'):s.evaluate(*inputs())
    assert len(calls)==before


def test_compiler_exception_is_not_successful_partial_population(monkeypatch):
    s,inputs,_,_=fixture(monkeypatch);cfg,compiled,ref=inputs()
    def broken():
        yield compiled[0]
        raise RuntimeError('audio compiler failed')
    with pytest.raises(RuntimeError):s.evaluate(cfg,broken(),ref)
    assert s.phase=='failed'
