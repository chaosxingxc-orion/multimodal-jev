import copy,importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from multimodal_jev.context_decision_head import ContextDecisionHead
from multimodal_jev.context_pooling import ContextHistoryReadout

spec=importlib.util.spec_from_file_location('context_policy_activation',Path(__file__).parents[1]/'tools/jev/context_policy_activation.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture():
    head=ContextDecisionHead(8,2,16).bfloat16().eval();embedding=torch.randn(20,8).bfloat16()
    contract=dict(token=dict(id=19),base_id='a'*64)
    current=dict(policy_id='b'*64,contract=contract,tensors={
        **{'decision_head.'+k:v.clone() for k,v in head.state_dict().items()},'decision_token_row':embedding[19].clone()})
    readout=ContextHistoryReadout(head,19,current['policy_id'])
    new=copy.deepcopy(current);new['policy_id']='c'*64
    new['tensors']['decision_head.projection.bias']+=.5
    return m.HeadActivation(readout,embedding,current),new


def evaluate(tx,policy):
    hidden=torch.randn(3,8).bfloat16();prompt=torch.tensor([1,2,19])
    return tx.readout(hidden,[3],[prompt],[dict(policy_id=policy,marker_positions=[2],expected_input_ids=prompt.tolist())])


def test_staged_head_rejects_old_and_new_requests_until_commit():
    tx,new=fixture();old=tx.current['policy_id'];evaluate(tx,old)
    receipt=tx.stage(new,expected_policy_id=old,slot=2)
    assert not receipt['serving_allowed']
    for policy in (old,new['policy_id']):
        with pytest.raises(ValueError):evaluate(tx,policy)
    tx.commit(dict(policy_id=new['policy_id'],slot=2,adapter_verified=True))
    evaluate(tx,new['policy_id'])
    with pytest.raises(ValueError):evaluate(tx,old)
    assert tx.generation==1


@pytest.mark.parametrize('fault',['marker','contract','head_shape','nan','current'])
def test_validation_failure_keeps_weights_and_closes_serving(fault):
    tx,new=fixture();before={k:v.clone() for k,v in tx.readout.head.state_dict().items()};old=tx.current['policy_id']
    if fault=='marker':new['tensors']['decision_token_row']+=1
    if fault=='contract':new['contract']['token']['id']=18
    if fault=='head_shape':new['tensors']['decision_head.projection.bias']=torch.zeros(2).bfloat16()
    if fault=='nan':new['tensors']['decision_head.projection.bias'][:]=float('nan')
    if fault=='current':old='wrong'
    with pytest.raises(ValueError):tx.stage(new,expected_policy_id=old,slot=2)
    assert tx.phase=='failed'
    assert all(torch.equal(v,tx.readout.head.state_dict()[k]) for k,v in before.items())
    with pytest.raises(ValueError):tx.stage(new,expected_policy_id=tx.current['policy_id'],slot=2)


def test_wrong_adapter_receipt_poisoned_and_cannot_retry():
    tx,new=fixture();tx.stage(new,expected_policy_id=tx.current['policy_id'],slot=2)
    with pytest.raises(ValueError):tx.commit(dict(policy_id=new['policy_id'],slot=1,adapter_verified=True))
    assert tx.phase=='failed'
    with pytest.raises(ValueError):tx.commit(dict(policy_id=new['policy_id'],slot=2,adapter_verified=True))


def test_a_b_a_switch_restores_exact_head():
    tx,new=fixture();original=copy.deepcopy(tx.current)
    for slot,bundle in ((2,new),(3,original)):
        tx.stage(bundle,expected_policy_id=tx.current['policy_id'],slot=slot)
        tx.commit(dict(policy_id=bundle['policy_id'],slot=slot,adapter_verified=True))
    assert tx.generation==2
    assert all(torch.equal(v,original['tensors']['decision_head.'+k]) for k,v in tx.readout.head.state_dict().items())
