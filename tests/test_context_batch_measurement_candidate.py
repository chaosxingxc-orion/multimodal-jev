import copy
from types import SimpleNamespace as NS

import pytest
import torch

from tools.reports.context_batch_measurement_candidate import install_counters, measure, compare
from test_context_batched_service_candidate import engine_case
from test_context_bulk_rpc_candidate import record


def measured(mode):
    engine,rows,calls=engine_case();groups=[]
    engine.score=lambda inputs,positions,**kw: (groups.append([inputs['input_ids'].shape[1]]) or
        record(engine.bundle['policy_id'],dict(inputs=inputs,positions=positions)))
    original=engine.score_many
    def score_many(batch,**kw):
        groups.append([r['inputs']['input_ids'].shape[1] for r in batch])
        return original(batch,**kw)
    engine.score_many=score_many
    def telemetry(op):
        if op=='reset':groups.clear();return {'reset':True}
        return dict(counts=dict(language=copy.deepcopy(groups),audio=[]),peak_allocated_bytes=10,peak_reserved_bytes=20)
    ticks=iter([0,2,3,5,6,8,9,11] if mode=='serial' else [0,1,2,3,4,5,6,7])
    result=measure(engine,rows,mode=mode,audio_token_id=98,telemetry=telemetry,clock=lambda:next(ticks))
    result['input_sha256']='a'*64
    return result


def test_comparison_binds_real_call_coverage_and_canonical_sampling():
    serial,batched=measured('serial'),measured('batched')
    r=compare(serial,batched)
    assert r['actual_language_calls']==[[2,2],[1,1]]
    assert r['time_ratio']==2 and r['actual_call_reduction']
    assert serial['passes'][3]['order']==[1,0]
    assert serial['passes'][0]['records']==serial['passes'][3]['records']


@pytest.mark.parametrize('fault',['probability','sampling','coverage','input','policy','missing_pass'])
def test_comparison_rejects_mismatch(fault):
    serial,batched=measured('serial'),measured('batched')
    if fault=='probability':batched['passes'][1]['records'][0]['prob'][0]=.9
    if fault=='sampling':batched['passes'][1]['sampling'][0]={}
    if fault=='coverage':batched['passes'][1]['telemetry']['counts']['language']=[[7]]
    if fault=='input':batched['input_sha256']='b'*64
    if fault=='policy':batched['policy_id']='b'*64
    if fault=='missing_pass':batched['passes'].pop()
    with pytest.raises(ValueError):compare(serial,batched)


def test_hooks_count_actual_first_attention_and_outer_audio_calls():
    class Layer(torch.nn.Module):
        def forward(self,positions,hidden):return hidden
    class Audio(torch.nn.Module):
        def forward(self,input_features,feature_lens,aftercnn_lens):return input_features
    layer=Layer();tower=Audio()
    model=NS(language_model=NS(model=NS(layers=[NS(self_attn=layer)])),audio_tower=tower)
    assert install_counters(model)['installed']
    layer(torch.tensor([0,1,0,1,2]),torch.ones(5,2))
    tower(torch.ones(4,8),feature_lens=torch.tensor([3,5]),aftercnn_lens=torch.tensor([1,1]))
    assert model._jev_measurement_counts==dict(language=[[2,3]],audio=[[3,5]])
    with pytest.raises(ValueError):install_counters(model)
    with pytest.raises(ValueError):layer(torch.tensor([0,2]),torch.ones(2,2))
