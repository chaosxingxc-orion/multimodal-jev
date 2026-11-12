import copy
import math
from pathlib import Path
import json
import pytest
from multimodal_jev.slurp_jev_inputs import compile_requests, decode_probabilities


def data():
    bank=dict(schema='slurp-jev-matched-ontology-v1',intents=[dict(id=f's_a{i}',scenario='s',action=f'a{i}') for i in range(60)],
              slot_types=[f't{i}' for i in range(55)],type_folds=[dict(fold_id=f'f{i}',types=[f't{j}' for j in range(i,55,10)]) for i in range(10)])
    defs=dict(schema='slurp-type-definition-candidates-v1',types=[dict(entity_type=t,definition='Definition '+t) for t in bank['slot_types']])
    request=dict(id='recording',official_split='train',role='learning',source_group='group',audio_sha256='a'*64,input=dict(audio='/audio.flac'))
    return request,bank,defs


def test_label_free_complete_requests_and_single_token_support():
    request,bank,defs=data();compiled=compile_requests(request,bank,defs)
    assert len(compiled)==11
    assert len(compiled[0]['request'].questions[0].values)==60
    assert [q.id for r in compiled[1:] for q in r['request'].questions]==[t for f in bank['type_folds'] for t in f['types']]
    for r in compiled:
        wire=json.dumps(r['request'].model_input())
        assert 'source_group' not in wire and 'audio_sha256' not in wire and 'targets' not in wire
    from multimodal_jev.span_contract import span_labels
    class Tokenizer:
        def encode(self,s,**kwargs):return [ord(s)]
    assert len(span_labels(Tokenizer(),compiled[0]['request'].questions)[0]['token_ids'])==60


@pytest.mark.parametrize('fault',['target','transcript','duplicate_fold','missing_type','missing_definition','alias'])
def test_rejects_contamination_and_incomplete_mapping(fault):
    r,b,d=data()
    if fault=='target':r['target']='s_a0'
    if fault=='transcript':r['input']['transcript']='words'
    if fault=='duplicate_fold':b['type_folds'][-1]=b['type_folds'][0]
    if fault=='missing_type':b['slot_types'].pop()
    if fault=='missing_definition':d['types'].pop()
    if fault=='alias':b['intents'][0]['id']='query'
    with pytest.raises(ValueError):compile_requests(r,b,d)


def test_decode_full_candidates_and_ties():
    compiled=compile_requests(*data())
    probs=[[[ -math.log(60)]*60]]+[[[math.log(.25),math.log(.75)] for q in r['request'].questions] for r in compiled[1:]]
    decoded,receipt=decode_probabilities(compiled,probs)
    assert decoded['intent_ranking']==[f's_a{i}' for i in range(60)]
    assert set(decoded['types'])=={f't{i}' for i in range(55)}
    assert receipt['tied_spans']==1
    with pytest.raises(ValueError):decode_probabilities(compiled,probs[:-1])
    probs[1][0]=[0.,0.]
    with pytest.raises(ValueError):decode_probabilities(compiled,probs)
