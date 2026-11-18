import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
from context_rollout_audit import verify_sample
from test_context_online_batches import Engine,item,m
from multimodal_jev.span_request_identity import tensor_identity


def evidence():
    query=item();query['positions']=torch.arange(1,61)[None]
    query['labels']=torch.zeros(1,60,dtype=torch.bool);query['labels'][0,0]=True
    query['observed']=torch.ones(1,60,dtype=torch.bool)
    ref=Engine('ref',[.7]*60);current=Engine('p0',[.4]*60)
    session=m.OnlineContextBatches(reference_engine=ref,reference_policy_id='ref',audio_token_id=3)
    prepared,_,receipts=session.prepare([query],current,policy_id='p0',update=0)
    return receipts[0],prepared[0]


def test_replay_uses_raw_probabilities_and_records_are_detached():
    entry,group=evidence()
    result=verify_sample(entry,group,update=0,policy_id='p0',reference_policy_id='ref')
    assert all(result.values())
    group['behavior'][0][0,1]=0
    assert entry['behavior_record']['prob'][0]==.4


@pytest.mark.parametrize('fault',['actions','behavior_log','reference_log','reference_policy','seed','input'])
def test_tampered_sampling_or_reference_is_rejected(fault):
    entry,group=copy.deepcopy(evidence())
    if fault=='actions':
        group['actions'][0,0,0]^=1
        # A self-consistent action hash must not hide incorrect sampling.
        entry['rollout']['actions']=tensor_identity(group['actions'])
    elif fault=='behavior_log':group['behavior'][0][0,1]+=.01
    elif fault=='reference_log':group['reference_policy'][0][0,1]+=.01
    elif fault=='reference_policy':entry['reference_policy_id']='other'
    elif fault=='seed':entry['rollout']['seed']+=1
    elif fault=='input':entry['reference_record']['request_identity']['positions'][0]+=1
    with pytest.raises(ValueError):
        verify_sample(entry,group,update=0,policy_id='p0',reference_policy_id='ref')
