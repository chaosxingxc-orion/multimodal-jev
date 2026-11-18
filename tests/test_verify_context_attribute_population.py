import copy
import json
from pathlib import Path
import pytest
from tools.reports.verify_context_attribute_population import inspect_records
from tools.reports.build_context_attribute_candidate import request,digest
from test_context_attribute_candidate import fixture
from tools.reports.build_context_attribute_candidate import catalog


def inputs_fixture():
    menu=catalog(fixture());anchors=[];arms=[]
    for arm in ('direct','q_input'):
        n=len(request('audio.wav',menu,arm=arm).questions)
        for samples,a in [(8191,7),(41979,34),(373760,304)]:
            anchors.append(dict(arm=arm,tokens=100+a,audio_tokens=a,positions=[[50+a+i for i in range(n)]]))
        arms.append(dict(arm=arm,tokens=107,audio_tokens=7,decisions=n,positions=[57+i for i in range(n)],
            request_sha256=digest(request('audio.wav',menu,arm=arm).model_input()),
            tensors={k:dict(shape=[1,107] if k in ('input_ids','attention_mask') else [1,128,52],sha256='a'*64)
                for k in ('input_ids','attention_mask','input_features','input_features_mask')}))
    metadata=[dict(id='a',samples=8191,audio_sha256='s',waveform_sha256='w')]
    inputs=[dict(id='a',audio='audio.wav',audio_sha256='s',catalog_sha256=digest(menu))]
    records=[dict(metadata[0],records=arms)]
    return records,inputs,metadata,menu,anchors


def test_independent_record_reconstruction():
    r=inspect_records(*inputs_fixture(),expected_count=1)
    assert r['requests']==2 and r['histogram']['q_input']=={'107':1}


@pytest.mark.parametrize('fault',['missing','position','waveform','request','feature','expansion','duplicate'])
def test_corruption_rejected(fault):
    records,inputs,metadata,menu,anchors=inputs_fixture()
    if fault=='missing':records=[]
    if fault=='position':records[0]['records'][0]['positions'][0]+=1
    if fault=='waveform':records[0]['waveform_sha256']='changed'
    if fault=='request':records[0]['records'][0]['request_sha256']='changed'
    if fault=='feature':records[0]['records'][0]['tensors']['input_features']['sha256']='b'*64
    if fault=='expansion':metadata[0]['samples']=16000;records[0]['samples']=16000
    if fault=='duplicate':inputs*=2
    with pytest.raises(ValueError):inspect_records(records,inputs,metadata,menu,anchors,expected_count=1)
