import copy
import pytest
from multimodal_jev.intent_correctness_replay import validate_factors,validate_rank


def test_relocated_explicit_default_is_same_objective():
    a=dict(arm='D',step=1,lr=1e-4,seed=317,previous_root='/old',pins={})
    b=dict(a,previous_root='/new',objective={'name':'correctness-v1'},adam_betas=[.9,.999])
    assert validate_factors(a,b)['passed']
    for key,value in [('seed',318),('unknown_factor',True),('adam_betas',[0,.999]),
                      ('objective',{'name':'smoothed-utility-v1'}),('step',0),('arm','Q')]:
        with pytest.raises(ValueError):validate_factors(a,dict(b,**{key:value}))


def test_rank_identity_cannot_be_substituted():
    a=dict(rank=0,world_size=4,updates=2,arm='D',base_id='base',initial_policy_id='initial',adapter_id='after',
        recording_ids=['a'],admission={'passed':True},execution_dtypes={'lora':'bf16'},
        frozen_base_unchanged=True,cross_rank_export_exact=True,actor_sha256='actor')
    validate_rank(a,copy.deepcopy(a),0)
    for key,value in [('initial_policy_id','other'),('recording_ids',['b']),('rank',1),('adapter_id','other')]:
        with pytest.raises(ValueError):validate_rank(a,dict(a,**{key:value}),0)
