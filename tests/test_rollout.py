from types import SimpleNamespace as NS
import pytest
from multimodal_jev.rollout import validate_sampling,decode_outputs


def config():return {'temperature':1.,'top_p':1.,'top_k':-1,'n':2,'max_tokens':16}


def test_sampling_parity_constraints():
    validate_sampling(config())
    for key,value in [('temperature',.8),('top_k',20),('n',1),('presence_penalty',1),('stop',['x']),('max_tokens',0)]:
        with pytest.raises(ValueError):validate_sampling({**config(),key:value})


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
