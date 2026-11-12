import copy
import hashlib
import json
import math
import pytest
from multimodal_jev.intent_gate_identity import verify_population


def rehash(row):
    row['request_id']=hashlib.sha256(json.dumps(row['request_identity'],sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()


def fixture():
    expected={'a::D':{'audio':'a.wav','questions':['final']},
              'b::Q':{'audio':'b.wav','questions':['process','final']}}
    sources=[];rows=[]
    for uid,request in expected.items():
        n=len(request['questions'])
        source=dict(id=uid,request=copy.deepcopy(request),positions=[list(range(n))],
            audio_processing={'waveform':'pinned'},candidate_mapping=[{'token_ids':[42,43]} for _ in range(n)],
            **{k:{'sha256':uid+k} for k in ('input_ids','input_features','input_features_mask')})
        identity=dict(version='jev-span-request-identity-v1',request=copy.deepcopy(request),
            policy_id='base:adapter',sampling={'temperature':1.0,'policy':'independent_finite_categorical'},
            audio_processing=source['audio_processing'],positions=source['positions'],
            candidate_ids=[m['token_ids'] for m in source['candidate_mapping']],
            inputs={k:source[k] for k in ('input_ids','input_features','input_features_mask')})
        row=dict(id=uid,request_identity=copy.deepcopy(identity),candidate_logprobs=[[-math.log(2)]*2 for _ in range(n)])
        rehash(row);sources.append(source);rows.append(row)
    return dict(passed=True,requests=2,records=sources),expected,rows


def test_full_population_not_just_training_subset():
    artifact,expected,rows=fixture()
    assert verify_population(artifact,expected,rows,'base:adapter')['requests']==2
    # Both backends could agree on this corrupted, self-consistently hashed row.
    rows[1]['request_identity']['request']['questions']=['wrong','final'];rehash(rows[1])
    with pytest.raises(ValueError,match='task identity'):verify_population(artifact,expected,rows,'base:adapter')


@pytest.mark.parametrize('field',['policy','support','tensor','audio','probability','duplicate','missing','hash'])
def test_authenticated_population_rejects_corruption(field):
    artifact,expected,rows=fixture();identity=rows[1]['request_identity']
    if field=='policy':identity['policy_id']='stale'
    if field=='support':identity['candidate_ids'][0].reverse()
    if field=='tensor':identity['inputs']['input_features']['sha256']='wrong'
    if field=='audio':identity['audio_processing']={'waveform':'different'}
    if field=='probability':rows[1]['candidate_logprobs'][0]=[0.,0.]
    rehash(rows[1])
    if field=='duplicate':rows.append(copy.deepcopy(rows[1]))
    if field=='missing':rows.pop()
    if field=='hash':rows[1]['request_id']='forged'
    with pytest.raises(ValueError):verify_population(artifact,expected,rows,'base:adapter')
