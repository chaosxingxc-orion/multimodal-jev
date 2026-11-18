import copy

import pytest
import torch

from multimodal_jev.context_logits_wire import WIRE
from multimodal_jev.context_rollout_batch import bind_rollout
from test_context_bulk_rpc_candidate import row, record
from tools.reports.context_macro_measurement import check_reference, measure


def fixture():
    rows=[row(3+i%5) for i in range(128)]
    for i,r in enumerate(rows):r['inputs']['input_features'].fill_(i)
    def records(batch):
        result=[record('p',r) for r in batch]
        for r in result:r.update(logits=torch.tensor(r['prob']).logit().tolist(),output_protocol=WIRE)
        return result
    values=records(rows)
    samples=[bind_rollout(r,**x,policy_id='p',group_size=16,seed=317+i)['receipt']
             for i,(r,x) in enumerate(zip(values,rows,strict=True))]
    reference=dict(mode='serial',rows=128,policy_id='p',input_sha256='digest',passes=[
        dict(kind=kind,records=copy.deepcopy(values),sampling=copy.deepcopy(samples))
        for kind in ('warmup','timed','timed','reversed')])
    class Pool:
        bundle={'policy_id':'p'};runtime_identity={'fixture':True}
        def score_many(self,batch,**kw):
            self.last_receipt=dict(completed_indices=list(range(128)),policy_id='p',
                                   status='complete_macro_inference')
            return records(batch)
    return rows,reference,Pool()


def test_four_macro_passes_compare_full_records_and_restore_reverse_order():
    rows,reference,pool=fixture()
    expected=check_reference(reference,rows,'p','digest')
    result=measure(pool,rows,expected,audio_token_id=1)
    assert result['exact_actual_logits'] and result['queries_per_second']>0
    assert len(result['passes'])==4
    assert all(p['records']==expected['records'] for p in result['passes'])


@pytest.mark.parametrize('fault',['short','sampling','logit','input','policy'])
def test_reference_is_independently_bound_before_service_calls(fault):
    rows,reference,_=fixture()
    p=reference['passes'][0]
    if fault=='short':p['records'].pop()
    if fault=='sampling':p['sampling'][0]['seed']+=1
    if fault=='logit':p['records'][0]['logits'][0]+=1
    if fault=='input':rows[0]['inputs']['input_features'].add_(1)
    if fault=='policy':reference['policy_id']='changed'
    with pytest.raises(ValueError):check_reference(reference,rows,'p','digest')


@pytest.mark.parametrize('fault',['short','reverse','changed_logit','receipt'])
def test_macro_corruption_never_reports_success(fault):
    rows,reference,pool=fixture();original=pool.score_many
    def corrupted(*args,**kw):
        records=original(*args,**kw)
        if fault=='short':records.pop()
        if fault=='reverse':records.reverse()
        if fault=='changed_logit':records[0]['logits'][0]+=1e-7
        if fault=='receipt':pool.last_receipt['completed_indices'].pop()
        return records
    pool.score_many=corrupted
    with pytest.raises(ValueError):measure(pool,rows,reference['passes'][0],audio_token_id=1)
