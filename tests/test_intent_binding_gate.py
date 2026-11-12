import copy
import hashlib
import json
import math
from dataclasses import replace
import pytest
from multimodal_jev.intent_binding_gate import verify_records
from multimodal_jev.intent_binding_training import expand
from multimodal_jev.span_candidate_intervention import from_input
from test_intent_binding_training import fixture as source_fixture


def rehash(row):
    row['request_id']=hashlib.sha256(json.dumps(row['request_identity'],sort_keys=True,separators=(',',':')).encode()).hexdigest()


def fixture():
    populations={'D':[],'Q':[]};cpu=[];records=[]
    for i in range(89):
        for arm in ('D','Q'):
            source=source_fixture();source['id']=str(i)
            if arm=='D':
                source['request']=replace(source['request'],questions=source['request'].questions[-1:])
                for k in ('targets','target_values','supervised','span_weights'):source['target'][k]=source['target'][k][-1:]
            for row,target in expand(source):
                request=from_input(row['request']);uid=str(i)+'::'+arm+'::'+row['variant']
                populations[arm].append(dict(row,request=request,target=target,gate_id=uid))
                mapping=[dict(values=list(q.values),token_ids=list(range(len(q.values)))) for q in request.questions]
                c=dict(id=uid,recording_id=str(i),arm=arm,variant=row['variant'],audio_sha256=row['audio_sha256'],
                    request=json.loads(json.dumps(request.model_input())),candidate_mapping=mapping,
                    positions=[list(range(len(mapping)))],audio_processing={'waveform':'synthetic'},
                    **{k:dict(sha256=uid+k) for k in ('input_ids','input_features','input_features_mask')})
                identity=dict(version='jev-span-request-identity-v1',request=c['request'],policy_id='base:adapter',
                    sampling=dict(temperature=1.0,policy='independent_finite_categorical'),audio_processing=c['audio_processing'],
                    positions=c['positions'],candidate_ids=[m['token_ids'] for m in mapping],
                    inputs={k:c[k] for k in ('input_ids','input_features','input_features_mask')})
                record=dict(id=uid,request_identity=copy.deepcopy(identity),candidate_logprobs=[[-math.log(len(m['values']))]*len(m['values']) for m in mapping])
                rehash(record);cpu.append(c);records.append(record)
    artifact=dict(passed=True,requests=712,recordings=89,records=cpu)
    hf=dict(passed=True,backend='hf',base_id='base',adapter_id='adapter',records=records)
    vl=copy.deepcopy(hf);vl['backend']='vllm'
    return populations,artifact,hf,vl


def test_full_population_verified_before_batch_selection():
    populations,artifact,hf,vl=fixture()
    indices,audit=verify_records(populations,artifact,hf,vl,'base:adapter')
    assert len(indices[0])==len(indices[1])==audit['requests']==712
    assert audit['maximum_logprob_error']==0


@pytest.mark.parametrize('defect',['missing','duplicate','native_only','both_wrong_task','both_wrong_policy',
    'both_wrong_tensor','both_wrong_values','audio','view','probability','support','parity','hash','backend','source'])
def test_rejects_corruption_even_outside_selected_batch(defect):
    populations,artifact,hf,vl=fixture()
    if defect=='missing':vl['records'].pop()
    elif defect=='duplicate':vl['records'].append(copy.deepcopy(vl['records'][-1]))
    elif defect=='native_only':vl['records']=[r for r in vl['records'] if r['id'].endswith('::native')]
    elif defect=='source':populations['Q'].pop()
    elif defect=='backend':vl['backend']='hf'
    elif defect=='audio':artifact['records'][-1]['audio_sha256']='different'
    elif defect=='view':artifact['records'][-1]['variant']='native'
    elif defect=='both_wrong_values':artifact['records'][-1]['candidate_mapping'][-1]['values'].reverse()
    elif defect.startswith('both_wrong'):
        for backend in (hf,vl):
            row=backend['records'][-1];identity=row['request_identity']
            if defect=='both_wrong_task':identity['request']['audio']='wrong.wav'
            elif defect=='both_wrong_policy':identity['policy_id']='stale'
            else:identity['inputs']['input_features']['sha256']='wrong'
            rehash(row)
    else:
        row=vl['records'][-1]
        if defect=='hash':row['request_id']='wrong'
        elif defect=='probability':row['candidate_logprobs'][-1][0]=0.
        elif defect=='support':row['candidate_logprobs'].pop()
        else:
            p=[1/61]*60;p[0]=2/61
            row['candidate_logprobs'][-1]=list(map(math.log,p))
    with pytest.raises(ValueError):verify_records(populations,artifact,hf,vl,'base:adapter')


def test_failed_full_gate_never_encodes_selected_audio(monkeypatch):
    import multimodal_jev.intent_binding_gate as module
    def failed(*args):raise ValueError('full gate rejected')
    def forbidden(*args):raise AssertionError('selection must not run')
    monkeypatch.setattr(module,'authenticate_artifacts',failed)
    monkeypatch.setattr(module,'select',forbidden)
    with pytest.raises(ValueError,match='full gate rejected'):module.prepare({},None,'p',0)
