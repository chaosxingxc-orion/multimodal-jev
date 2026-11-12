import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import pytest

TOOLS = Path(__file__).resolve().parents[1]/'tools/jev'
sys.path.insert(0, str(TOOLS))
from score_intent_binding_matrix import validate_records


def fixture():
    row = dict(id='source::D::native', request={'audio':'a.wav'}, audio_processing={'hash':'a'}, positions=[[10]],
        candidate_mapping=[dict(token_ids=[1,2], values=['a','b'])], input_ids={'sha256':'x'},
        input_features={'sha256':'y'}, input_features_mask={'sha256':'z'})
    identity = dict(version='jev-span-request-identity-v1', request=row['request'], audio_processing=row['audio_processing'],
        positions=row['positions'], candidate_ids=[[1,2]], policy_id='base:adapter',
        sampling=dict(temperature=1.0,policy='independent_finite_categorical'),
        inputs={k:row[k] for k in ('input_ids','input_features','input_features_mask')})
    digest = hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    record = dict(id=row['id'],request_id=digest,request_identity=identity,candidate_logprobs=[[math.log(.7),math.log(.3)]])
    hf = dict(passed=True,backend='hf',base_id='base',adapter_id='adapter',records=[record])
    vl = copy.deepcopy(hf);vl['backend']='vllm'
    return dict(records=[row]),hf,vl


def test_content_verification_accepts_identical_full_policy():
    inputs,hf,vl=fixture()
    rows,audit=validate_records(inputs,hf,vl,'base','adapter')
    assert len(rows)==1 and audit['maximum_logprob_error']==0


@pytest.mark.parametrize('mutation', ['duplicate','missing','identity','probability','parity','policy','support'])
def test_content_verification_rejects_false_readiness(mutation):
    inputs,hf,vl=fixture()
    if mutation=='duplicate': vl['records']*=2
    if mutation=='missing': vl['records']=[]
    if mutation=='identity': vl['records'][0]['request_identity']['policy_id']='other'
    if mutation=='probability': vl['records'][0]['candidate_logprobs']=[[0.,0.]]
    if mutation=='parity': vl['records'][0]['candidate_logprobs']=[[math.log(.6),math.log(.4)]]
    if mutation=='policy': vl['adapter_id']='different'
    if mutation=='support': vl['records'][0]['candidate_logprobs']=[]
    with pytest.raises(ValueError): validate_records(inputs,hf,vl,'base','adapter')
