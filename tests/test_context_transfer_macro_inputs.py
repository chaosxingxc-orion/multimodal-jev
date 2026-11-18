import copy
import pytest
import torch

from multimodal_jev.context_candidate_task import compile_candidates
from multimodal_jev.context_decision_contract import render_context_request
from multimodal_jev.slurp_training import content_hash
from multimodal_jev.span_request_identity import tensor_identity
from tools.reports.context_transfer_macro_inputs import normalize


def fixture(count):
    candidates=[dict(id=str(i),definition=f'candidate {i}') for i in range(count)]
    requests=[compile_candidates(f'/audio/{i}.wav',candidates,['listen'],semantics='exclusive_event') for i in range(3)]
    rows=[];receipts=[]
    for i,request in enumerate(requests):
        system,text=render_context_request(request)
        row=dict(id=str(i),audio_sha256='audio',request_sha256=content_hash(request.model_input()),
            rendered_sha256=content_hash(dict(system=system,text=text)))
        tensors={k:dict(shape=[1],dtype='torch.int64',sha256='fixture') for k in
            ('input_ids','attention_mask','input_features','input_features_mask')}
        tensors['positions']=tensor_identity(torch.arange(count).reshape(1,count))
        rows.append(row);receipts.append(dict(row,positions=[list(range(count))],tensors=tensors))
    return rows,receipts,requests,[str(i) for i in range(count)]


@pytest.mark.parametrize('count',[14,31])
def test_preserves_complete_target_menu_and_drops_non_input_metadata(count):
    rows,receipts,requests,ids=fixture(count)
    for row in rows:row.update(speaker_id='speaker',language='lang')
    population,independent=normalize(rows,receipts,requests,ids,expected_count=3,decisions=count)
    assert len(population)==len(independent)==3
    assert all(set(p)=={'id','audio_sha256','request_sha256'} for p in population)
    assert all(r['candidate_ids']==ids and 'audio_receipt' not in r for r in independent)


@pytest.mark.parametrize('fault',['missing','duplicate','order','candidate','position','tensor','audio','request','rendered'])
def test_corrupt_target_binding_refused(fault):
    rows,receipts,requests,ids=fixture(31)
    if fault=='missing':receipts.pop()
    if fault=='duplicate':rows[1]=copy.deepcopy(rows[0])
    if fault=='order':receipts.reverse()
    if fault=='candidate':ids.reverse()
    if fault=='position':receipts[0]['positions'][0][0]=1
    if fault=='tensor':receipts[0]['tensors']['positions']['sha256']='other'
    if fault=='audio':receipts[0]['audio_sha256']='other'
    if fault=='request':rows[0]['request_sha256']='other'
    if fault=='rendered':rows[0]['rendered_sha256']='other'
    with pytest.raises(ValueError):normalize(rows,receipts,requests,ids,expected_count=3,decisions=31)
