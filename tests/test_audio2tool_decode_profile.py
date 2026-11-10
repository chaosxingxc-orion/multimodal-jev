import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from multimodal_jev.audio2tool_decode_profile import profile_case,summarize
from multimodal_jev.audio2tool_inference import digest


def setup(family='native',fault=None):
    ids=[4,5,6];count=1024 if fault=='length' else 2
    raw='not JSON'
    class Processor:
        tokenizer=SimpleNamespace(decode=lambda *a,**kw:'different' if fault=='decode' else raw)
        def __call__(self,**kw):
            assert kw['truncation'] is False
            return dict(input_ids=torch.tensor([ids if fault!='input_ids' else [4,6,5]]),
                feature_attention_mask=torch.tensor([[1,1 if fault!='frames' else 0]]),
                input_features=torch.zeros(1,2,2))
    class Engine:
        calls=0
        def generate(self,*a,**kw):
            self.calls+=1
            if fault=='runtime':raise RuntimeError('actual engine failure')
            return [SimpleNamespace(finished=True,prompt_token_ids=ids,outputs=[SimpleNamespace(
                token_ids=[7]*count,text=raw,finish_reason='length' if fault=='length' else 'stop')])]
    case=dict(name=family+'-boundary',family=family,id='source',prior=dict(
        input_tokens=3,input_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest() if family=='native' else digest(ids),frames=[2]))
    def prepare():return dict(prompt='audio task',multi_modal_data={'audio':[([0.,0.],16000)]}),[{'expected_frames':2}]
    return Processor(),Engine(),case,prepare


@pytest.mark.parametrize('family',['native','jev'])
@pytest.mark.parametrize('fault',[None,'length'])
def test_original_hash_convention_raw_output_and_no_replay(tmp_path,family,fault):
    p,e,c,prepare=setup(family,fault)
    r=profile_case(p,e,None,c,prepare,tmp_path,lambda _:True)
    assert r['response']['raw_text']=='not JSON'
    assert r['response']['length_limited'] is (fault=='length')
    assert e.calls==1
    with pytest.raises(FileExistsError):profile_case(p,e,None,c,prepare,tmp_path,lambda _:True)
    assert e.calls==1


@pytest.mark.parametrize('fault',['input_ids','frames','runtime','decode'])
def test_actual_failure_preserved_without_fabricated_success(tmp_path,fault):
    p,e,c,prepare=setup(fault=fault)
    with pytest.raises((ValueError,RuntimeError)):profile_case(p,e,None,c,prepare,tmp_path,lambda _:True)
    if fault in ('input_ids','frames'):assert e.calls==0
    else:
        r=json.loads((tmp_path/(c['name']+'.json')).read_text());assert r['error'] and e.calls==1
        if fault=='decode':assert r['response']['raw_text']=='not JSON'
        with pytest.raises(FileExistsError):profile_case(p,e,None,c,prepare,tmp_path,lambda _:True)
        assert e.calls==1


def test_preflight_cannot_claim_inference_timing(tmp_path):
    p,e,c,prepare=setup()
    r=profile_case(p,None,None,c,prepare,tmp_path,lambda _:True)
    assert r['model_loaded'] is False and 'response' not in r and e.calls==0


def records():
    return [dict(name=str(i),family='native' if i<4 else 'jev',model_loaded=True,input_tokens=10,
        preparation_seconds=.2,engine_seconds=2.,response=dict(token_ids=[7,8],length_limited=False)) for i in range(16)]


def test_boundary_summary_never_extrapolates_population():
    rows=records();r=summarize(rows,[x['name'] for x in rows])
    assert r['population_throughput_estimate'] is None
    assert r['boundary_observations']['native']['cases']==4
    assert r['boundary_observations']['jev']['engine_seconds']==24.


@pytest.mark.parametrize('fault',['missing','order','preflight','duration','error'])
def test_incomplete_or_invalid_profile_cannot_be_summarized(fault):
    rows=records();names=[x['name'] for x in rows]
    if fault=='missing':rows.pop()
    if fault=='order':rows.reverse()
    if fault=='preflight':rows[0]['model_loaded']=False
    if fault=='duration':rows[0]['engine_seconds']=float('nan')
    if fault=='error':rows[0]['error']='runtime failed'
    with pytest.raises(ValueError):summarize(rows,names)
