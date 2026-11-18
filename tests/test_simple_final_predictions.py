import copy
import hashlib
import json
import math
import pytest
from multimodal_jev.simple_final_predictions import from_backend


def fixture():
    mapping = dict(values=[str(i) for i in range(60)], token_ids=list(range(60)))
    row = dict(id='audio1', recording_id='audio1', view='direct', variant='native',
               request={'audio': '/a.wav'}, audio_sha256='filehash', audio_processing={'wave': 'h'},
               positions=[[42]], candidate_mapping=[mapping], input_ids={'sha256': 'tokens'},
               input_features={'sha256': 'features'}, input_features_mask={'sha256': 'mask'})
    identity = dict(policy_id='b:a', request=row['request'], audio_processing=row['audio_processing'],
                    positions=row['positions'], candidate_ids=[mapping['token_ids']],
                    inputs={key: row[key] for key in ('input_ids','input_features','input_features_mask')})
    request_id = hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    output = dict(id='audio1', request_identity=identity, request_id=request_id,
                  candidate_logprobs=[[-math.log(60)]*60])
    return dict(passed=True,population_mode='full_selection',records=[row],recordings=1,requests=1), dict(
        passed=True,backend='hf',base_id='b',adapter_id='a',records=[output])


def test_exact_backend_input_binding():
    artifact, result = fixture()
    prediction = from_backend(artifact,result,policy_id='b:a')[0]
    assert prediction['prediction'] == '0'
    assert prediction['candidate_values'] == artifact['records'][0]['candidate_mapping'][0]['values']


@pytest.mark.parametrize('fault',['missing','duplicate','policy','audio','mapping','features','request_hash','prefix'])
def test_reject_stale_or_partial_backend(fault):
    artifact,result = copy.deepcopy(fixture())
    if fault == 'missing': result['records']=[]
    if fault == 'duplicate': result['records']*=2
    if fault == 'policy': result['adapter_id']='stale'
    if fault == 'audio': result['records'][0]['request_identity']['request']={'audio':'/other.wav'}
    if fault == 'mapping': artifact['records'][0]['candidate_mapping'][0]['token_ids']=list(reversed(range(60)))
    if fault == 'features': artifact['records'][0]['input_features']={'sha256':'changed'}
    if fault == 'request_hash': result['records'][0]['request_id']='fake'
    if fault == 'prefix': artifact['population_mode']='prefix'
    with pytest.raises(ValueError): from_backend(artifact,result,policy_id='b:a')
