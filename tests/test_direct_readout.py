from dataclasses import replace
import pytest
from multimodal_jev.direct_readout import alternatives,prompt,label_tokens,decision
from multimodal_jev.consumption import request_from_example
from multimodal_jev.task_schema import Candidate,DecisionSchema,JevExample


def example():
    return JevExample('private:noul.True','train','/audio',
        DecisionSchema('schema','choice','Which command?',(Candidate('a','on'),Candidate('b','off'))),
        'a',{'source_group':'private','audio_sha256':'hash'})


def test_rebinding_and_permutation_follow_definitions_not_fixed_value():
    ex=example();request=request_from_example(ex);options=alternatives(request)
    assert [o['value'] for o in options]==['a','b',None]
    assert decision(options,[3,1,0])['value']=='a'
    rebound=replace(ex,schema=replace(ex.schema,candidates=(Candidate('b','on'),Candidate('a','off'))),target='b')
    assert decision(alternatives(request_from_example(rebound)),[3,1,0])['value']=='b'
    perm=replace(ex,schema=replace(ex.schema,candidates=tuple(reversed(ex.schema.candidates))))
    assert decision(alternatives(request_from_example(perm)),[1,3,0])['value']=='a'
    text,_=prompt(request,'listen')
    assert 'private' not in text and '/audio' not in text and 'noul.True' not in text


def test_noul_values_keep_types_and_unknown_is_optional():
    ex=replace(example(),schema=DecisionSchema('n','noul','Is it on?',allow_unknown=False),target=True)
    options=alternatives(request_from_example(ex))
    assert decision(options,[1,2])['value'] is True
    assert decision(options,[2,1])['value'] is False
    assert len(options)==2
    ex=replace(ex,schema=replace(ex.schema,allow_unknown=True))
    assert decision(alternatives(request_from_example(ex)),[0,0,1])['value'] is None


def test_single_token_and_finite_logit_checks():
    options=alternatives(request_from_example(example()))
    class Tokenizer:
        def encode(self,text,**kwargs):return [ord(text)]
        def decode(self,ids,**kwargs):return chr(ids[0])
    assert label_tokens(Tokenizer(),options)==[48,49,50]
    class Split(Tokenizer):
        def encode(self,text,**kwargs):return [1,2]
    with pytest.raises(ValueError,match='single|exactly one'):label_tokens(Split(),options)
    for logits in [[float('nan'),0,1],[1,2],[]]:
        with pytest.raises(ValueError):decision(options,logits)
    assert sum(decision(options,[10000,9999,-10000])['probabilities'])==pytest.approx(1.)


def test_open_slots_not_recast_as_closed_choices():
    ex=replace(example(),schema=DecisionSchema('s','open_slot','Who?'),target='Amy')
    with pytest.raises(ValueError,match='open_slot'):alternatives(request_from_example(ex))
