import copy
import pytest
torch=pytest.importorskip('torch')
from multimodal_jev.simple_resident_acceptance import compare_updated_batch
from multimodal_jev.simple_audio_batch import renderer_for_policy
from test_simple_resident_batch import setup


def fixture(monkeypatch):
    c,_,slots,config,_,_=setup(monkeypatch)
    slots[65:]=[dict(slots[0],padding=True,weight=0.) for _ in range(63)]
    items,receipt=c.collect(slots,policy_config=config,completed_updates=1)
    rendered=renderer_for_policy(c.last_renderer,'base:updated')
    result=c.engine.rollout(dict(config,adapter_id='updated',population={'requests':len(rendered)}),rendered)
    result['reference_compared']=False
    by_id={r['request_id']:r for r in result['records']}
    mapping={a[4]:b[4] for a,b in zip(c.last_renderer,rendered)}
    resident=[dict(request_id=item['request_id'],candidate_logprobs=copy.deepcopy(
        by_id[mapping[sample['request_id']]]['candidate_logprobs']))
        for item,sample in zip(items,receipt['sampling']['sampling'])]
    return [resident[i:i+32] for i in range(0,128,32)],result,receipt['sampling']


def test_all_global_slots_and_duplicate_padding_inputs_are_distinct(monkeypatch):
    ranks,result,sampling=fixture(monkeypatch)
    check=compare_updated_batch(ranks,result,sampling,policy_id='base:updated',updates=2)
    assert check['requests']==128 and check['unique_inputs']==65 and check['maximum_logprob_error']==0


@pytest.mark.parametrize('fault',['duplicate_slot','missing_rank','duplicate_backend','foreign_policy',
    'input_hash','wrong_slot','wrong_cursor','probability','extra_input'])
def test_reject_false_success_and_wrong_coverage(monkeypatch,fault):
    ranks,result,sampling=fixture(monkeypatch)
    if fault=='duplicate_slot':ranks[3][-1]=copy.deepcopy(ranks[0][0])
    elif fault=='missing_rank':ranks.pop()
    elif fault=='duplicate_backend':result['records'].append(copy.deepcopy(result['records'][0]))
    elif fault=='foreign_policy':result['adapter_id']='other'
    elif fault=='input_hash':result['records'][0]['request_identity']['request']['audio']='other'
    elif fault=='wrong_slot':sampling['sampling'][0]['slot']=1
    elif fault=='wrong_cursor':sampling['update']=0
    elif fault=='probability':ranks[0][0]['candidate_logprobs'][0][0]+=.1
    else:result['records'].pop()
    with pytest.raises(ValueError):compare_updated_batch(ranks,result,sampling,policy_id='base:updated',updates=2)
