"""Run in fixed torch CPU image; local project Python need not include torch."""
import copy
import pytest
import torch
from multimodal_jev.span_smoothed_utility import transform,utility_credit
from multimodal_jev.intent_objective import apply


def row():
    return dict(request_id='recording',behavior=(torch.tensor([[.001,.999]]).log(),),
        targets=torch.tensor([[0]]),supervised=torch.tensor([[True]]),
        span_weights=torch.ones(1,1),request_weight=2.,inputs={'audio':'identity'})


def test_transform_preserves_input_behavior_and_outer_weights():
    a=row();b,receipt=transform([a]);b=b[0]
    assert b['inputs'] is a['inputs'] and b['behavior'] is a['behavior']
    assert b['request_weight']==2 and a['span_weights'].item()==1
    assert b['span_weights'].item()==pytest.approx(utility_credit(.001,2)['scale'])
    assert not b['span_weights'].requires_grad
    for name in ('correctness-v1','uniform-credit-v1','smoothed-utility-v1'):
        objective={'name':name}
        if name=='uniform-credit-v1':objective['scale']=6
        result,receipt=apply(dict(arm='D',objective=objective),[a])
        assert receipt['objective']['specification']['name']==name
        if name=='uniform-credit-v1':assert result[0]['span_weights'].item()==6


def test_missing_label_never_becomes_supervision():
    a=row();a['supervised'].fill_(False);a['targets'].fill_(-100)
    out,receipt=transform([a])
    assert not out[0]['supervised'].item() and out[0]['span_weights'].item()==1
    assert receipt['records'][0]['credit'] is None


@pytest.mark.parametrize('name', ['correctness-v1','uniform-credit-v1','smoothed-utility-v1'])
def test_independent_credit_audit_rejects_altered_rewards_and_old_policy(name):
    from multimodal_jev.intent_objective import verify_credit,supervision_records
    a=row();objective={'name':name}
    if name=='uniform-credit-v1':objective['scale']=6
    cfg=dict(arm='D',objective=objective)
    out,receipt=apply(cfg,[a]);supervision=supervision_records(out)
    selected=[dict(id='source',target=dict(targets=[0],supervised=[True],span_weights=[1.],request_weight=2.))]
    behavior=dict(records=[dict(id='source::D',request_id=a['request_id'],candidate_logprobs=[a['behavior'][0][0].tolist()])])
    verify_credit(cfg,selected,behavior,receipt,supervision)
    corrupted=copy.deepcopy(supervision);corrupted[0]['span_weights'][0][0]+=1
    with pytest.raises(ValueError,match='weights'):verify_credit(cfg,selected,behavior,receipt,corrupted)
    corrupted=copy.deepcopy(receipt);corrupted['objective']['sha256']='changed'
    with pytest.raises(ValueError,match='receipt'):verify_credit(cfg,selected,behavior,corrupted,supervision)
    if name=='smoothed-utility-v1':
        behavior['records'][0]['candidate_logprobs']=[torch.tensor([.5,.5]).log().tolist()]
        with pytest.raises(ValueError,match='receipt'):verify_credit(cfg,selected,behavior,receipt,supervision)


@pytest.mark.parametrize('change', ['duplicate','old_grad','unnormalized','target','weight',
    'weight_grad','multi_span','negative_request','nan_request','bool_request','wrong_target_dtype'])
def test_reject_bad_credit_inputs(change):
    a=row();rows=[a]
    if change=='duplicate':rows.append(copy.deepcopy(a))
    elif change=='old_grad':a['behavior'][0].requires_grad_()
    elif change=='unnormalized':a['behavior']=(torch.zeros(1,2),)
    elif change=='target':a['targets'].fill_(2)
    elif change=='weight':a['span_weights'].fill_(.1)
    elif change=='weight_grad':a['span_weights'].requires_grad_()
    elif change=='multi_span':a['behavior']=a['behavior']*2
    elif change=='negative_request':a['request_weight']=-1
    elif change=='nan_request':a['request_weight']=float('nan')
    elif change=='bool_request':a['request_weight']=True
    elif change=='wrong_target_dtype':a['targets']=a['targets'].float()
    with pytest.raises(ValueError):transform(rows)
