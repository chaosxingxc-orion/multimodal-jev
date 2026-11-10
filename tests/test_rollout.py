from types import SimpleNamespace as NS
import pytest
from multimodal_jev.rollout import validate_sampling,decode_outputs


def config():return {'temperature':1.,'top_p':1.,'top_k':-1,'n':2,'max_tokens':16}


def test_sampling_parity_constraints():
    validate_sampling(config())
    for key,value in [('temperature',.8),('top_k',20),('n',1),('presence_penalty',1),('stop',['x']),('max_tokens',0)]:
        with pytest.raises(ValueError):validate_sampling({**config(),key:value})


def test_engine_disables_checkpoint_generation_defaults(monkeypatch):
    import sys
    from multimodal_jev.rollout import VllmAudioRollout
    received={}
    def fake_llm(**kwargs):received.update(kwargs);return NS()
    monkeypatch.setitem(sys.modules,'vllm',NS(LLM=fake_llm))
    VllmAudioRollout('checkpoint',NS(),'policy',{'dtype':'float32'})
    assert received['generation_config']=='vllm'
    assert received['limit_mm_per_prompt']=={'audio':1}
    assert received['enable_mm_embeds'] is False


def test_feature_engine_admission_and_mode_consistency(monkeypatch):
    import sys
    from multimodal_jev.rollout import VllmAudioRollout
    received={}
    def fake_llm(**kwargs):received.update(kwargs);return NS()
    monkeypatch.setitem(sys.modules,'vllm',NS(LLM=fake_llm,SamplingParams=object))
    rollout=VllmAudioRollout('checkpoint',NS(),'policy',{},'actor_features')
    assert received['enable_mm_embeds'] is True
    with pytest.raises(ValueError,match='initialization'):
        rollout.sample([],[],config(),'waveform')
    with pytest.raises(ValueError,match='unsupported'):
        VllmAudioRollout('checkpoint',NS(),'policy',{},'unknown')


def outputs():
    return [NS(finished=True,outputs=[NS(index=i,token_ids=[4,5],logprobs=[{4:NS(logprob=-.2)},{5:NS(logprob=-.3)}],finish_reason='stop') for i in [1,0]])]


def test_real_logprob_contract_and_replicate_order():
    processor=NS(tokenizer=NS(decode=lambda ids,**kwargs:'decoded'))
    rows=decode_outputs(outputs(),[NS(example_id='a')],processor,'checkpoint-sha',2)
    assert [r.replicate for r in rows]==[0,1]
    assert rows[0].log_probs==(-.2,-.3)
    assert rows[0].policy_id=='checkpoint-sha'
    broken=outputs();broken[0].outputs[0].logprobs[0]={}
    with pytest.raises(ValueError,match='missing'):decode_outputs(broken,[NS(example_id='a')],processor,'p',2)
    broken=outputs();broken[0].outputs[0].index=0
    with pytest.raises(ValueError,match='replicate'):decode_outputs(broken,[NS(example_id='a')],processor,'p',2)


def test_policy_update_rejects_stale_or_incomplete_rollouts():
    from multimodal_jev.rl_update import validate_rollouts
    processor=NS(tokenizer=NS(decode=lambda ids,**kwargs:'decoded'))
    examples=[NS(example_id='a')]
    records=decode_outputs(outputs(),examples,processor,'policy-1',2)
    validate_rollouts(examples,records,'policy-1',2)
    with pytest.raises(ValueError,match='stale'):validate_rollouts(examples,records,'policy-2',2)
    with pytest.raises(ValueError,match='incomplete'):validate_rollouts(examples,records[:1],'policy-1',2)
    with pytest.raises(ValueError,match='duplicate'):validate_rollouts(examples,records+records,'policy-1',2)


def test_generation_counts_engine_prompts_and_variable_response_lengths():
    from multimodal_jev.rollout import generation_budget
    requests=[NS(prompt_token_ids=[1]*7),NS(prompt_token_ids=[1]*11)]
    records=[NS(token_ids=[1]*length,finish_reason=reason)
             for length,reason in [(2,'stop'),(4,'length'),(3,'stop'),(5,'length')]]
    budget=generation_budget(requests,records,2,1.25)
    assert budget['request_prompt_tokens']==18
    assert budget['replicated_prompt_tokens']==36
    assert budget['generated_tokens']==14 and budget['length_terminated_responses']==2
    assert budget['prompt_tokens_per_request']==[7,11]
    assert budget['response_tokens_per_record']==[2,4,3,5]
    assert budget['engine_generate_seconds']==1.25
    with pytest.raises(ValueError,match='prompt token'):
        generation_budget([NS(),requests[1]],records,2,1.25)
    with pytest.raises(ValueError,match='complete'):
        generation_budget(requests,records[:-1],2,1.25)
    with pytest.raises(ValueError,match='duration'):
        generation_budget(requests,records,2,float('nan'))


def test_sampling_records_engine_accounting_and_clears_previous_result(monkeypatch):
    import sys
    import multimodal_jev.rollout as module
    returned=outputs();returned[0].prompt_token_ids=[1,2,3]
    engine=NS(generate=lambda *a,**k:returned)
    monkeypatch.setitem(sys.modules,'vllm',NS(LLM=lambda **k:engine,SamplingParams=lambda **k:k))
    monkeypatch.setattr(module,'vllm_request',lambda *a:{'prompt':'test'})
    prompt=[1,2,3]
    monkeypatch.setattr(module,'hf_prompt',lambda *a:{'input_ids':[NS(tolist=lambda:list(prompt))]})
    processor=NS(tokenizer=NS(decode=lambda ids,**k:'decoded'))
    runner=module.VllmAudioRollout('checkpoint',processor,'policy',{})
    records=runner.sample([NS(example_id='a')],[object()],config())
    assert len(records)==2 and runner.last_generation_budget['generated_tokens']==4
    assert runner.last_generation_budget['request_prompt_tokens']==3
    assert runner.last_prompt_identity==[{'example_id':'a','prompt_token_ids':[1,2,3]}]
    # Equal lengths cannot detect changed token identities.
    prompt[1]=99
    with pytest.raises(ValueError,match='prompt IDs'):
        runner.sample([NS(example_id='a')],[object()],config())
    assert runner.last_generation_budget is None and runner.last_prompt_identity is None
    prompt[1]=2
    returned[0].finished=False
    with pytest.raises(ValueError,match='finished'):
        runner.sample([NS(example_id='a')],[object()],config())
    assert runner.last_generation_budget is None
    runner.max_model_len=18
    returned[0].finished=True
    with pytest.raises(ValueError,match='exceed context'):
        runner.sample([NS(example_id='a')],[object()],config())


def test_explicit_shutdown_once_and_reject_reuse(monkeypatch):
    import sys
    from multimodal_jev.rollout import VllmAudioRollout
    calls=[]
    engine=NS(llm_engine=NS(engine_core=NS(shutdown=lambda **kw:calls.append(kw))))
    monkeypatch.setitem(sys.modules,'vllm',NS(LLM=lambda **kw:engine,SamplingParams=object))
    runner=VllmAudioRollout('checkpoint',NS(),'policy',{})
    runner.close();runner.close()
    assert calls==[{'timeout':30.0}]
    with pytest.raises(RuntimeError,match='closed'):
        runner.sample([],[],config())
