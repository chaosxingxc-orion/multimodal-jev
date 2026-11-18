import copy
import json

import pytest
import torch

from test_audit_context_macro_probe import build
from tools.reports.audit_context_macro_training import audit_journal,activation,prefix_mask_count
from multimodal_jev.span_request_identity import tensor_identity


def test_full_training_journal_is_independently_bound(tmp_path,monkeypatch):
    package,collection=build(tmp_path,monkeypatch)
    c=json.loads((package/'measurement.json').read_text())['macro_client'];c['max_queries']=128
    records=json.loads((collection/'measurement/measurement.json').read_text())['passes'][0]['records']
    d=collection/'journal/attempt/batch-00000000'
    assert audit_journal(d,records,policy='p',client=c)['rows']==128
    records[0]['request_identity']['inputs']['input_ids']['sha256']='0'*64
    with pytest.raises(ValueError,match='request identity'):audit_journal(d,records,policy='p',client=c)


@pytest.mark.parametrize('fault',['owner','complete','failed','membership'])
def test_journal_rejects_false_completion(tmp_path,monkeypatch,fault):
    package,collection=build(tmp_path,monkeypatch)
    c=json.loads((package/'measurement.json').read_text())['macro_client'];c['max_queries']=128
    records=json.loads((collection/'measurement/measurement.json').read_text())['passes'][0]['records']
    d=collection/'journal/attempt/batch-00000000'
    p=d/'rollout-0000.json'
    if fault=='complete':p=d/'COMPLETE.json'
    r=json.loads(p.read_text())
    if fault=='owner':r['replica_id']=1-r['replica_id']
    if fault=='complete':r['completed_indices'].pop()
    if fault=='membership':r['records'].reverse()
    if fault=='failed':(d/'FAILED.json').write_text('{}')
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError):audit_journal(d,records,policy='p',client=c)


def test_mask_cost_uses_hash_verified_valid_prefix_not_padded_width():
    mask=torch.cat((torch.ones(1,211,dtype=torch.long),torch.zeros(1,89,dtype=torch.long)),dim=1)
    assert prefix_mask_count(json.dumps(tensor_identity(mask),sort_keys=True))==211
    mask[0,2]=0
    with pytest.raises(ValueError,match='mask hash'):prefix_mask_count(json.dumps(tensor_identity(mask),sort_keys=True))


def test_both_replica_receipts_required():
    r=dict(policy_id='p',head_marker_exact=True,native_loaded_tensors=112)
    receipt=dict(r,replica_receipts=[copy.deepcopy(r),copy.deepcopy(r)])
    activation(receipt,'p')
    receipt['replica_receipts'][1]['policy_id']='stale'
    with pytest.raises(ValueError,match='individual replica'):activation(receipt,'p')


def test_reference_is_immutable_and_requires_exact_roundtrip_activation():
    from tools.reports.audit_context_macro_training import reference_transition
    def ack(policy):
        r=dict(policy_id=policy,head_marker_exact=True,native_loaded_tensors=112)
        return dict(r,replica_receipts=[copy.deepcopy(r),copy.deepcopy(r)])
    reference=dict(request_id='input',prob=[.2,.8])
    receipt=dict(queries=[dict(reference_record=reference)],reference_misses=1,
        bulk_reference_calls=1,bulk_behavior_calls=1,switches=[ack('ref'),ack('new')])
    seen={}
    assert reference_transition(receipt,seen,current='new',reference='ref')==[reference]
    hit=copy.deepcopy(receipt);hit.update(reference_misses=0,bulk_reference_calls=0,switches=[])
    assert reference_transition(hit,seen,current='new',reference='ref')==[]
    hit['queries'][0]['reference_record']['prob'][0]=.3
    with pytest.raises(ValueError,match='fixed reference changed'):
        reference_transition(hit,seen,current='new',reference='ref')
    receipt['switches'].reverse()
    with pytest.raises(ValueError,match='activation sequence'):
        reference_transition(receipt,{},current='new',reference='ref')
