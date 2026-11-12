from itertools import product
import pytest
import torch
from multimodal_jev.span_candidate_transport import unpack_candidate_logits
from multimodal_jev.span_parity import compare_span_policies


def test_sixty_way_and_six_binary_complete_policy():
    logits=torch.arange(60,dtype=torch.float32)[None]/10
    result=unpack_candidate_logits(logits,[list(range(60))])
    assert torch.equal(result[0],logits[0].log_softmax(0))
    binary=torch.tensor([[.1,.2]]*6)
    a=unpack_candidate_logits(binary,[[32,33]]*6)
    b=unpack_candidate_logits(binary+torch.tensor([.001,-.001]),[[32,33]]*6)
    report=compare_span_policies(a,b)
    ratios=[torch.exp(sum(a[j][i]-b[j][i] for j,i in enumerate(choice))).item() for choice in product(range(2),repeat=6)]
    assert report['joint_ratio_min']==pytest.approx(min(ratios),abs=1e-6)
    assert report['joint_ratio_max']==pytest.approx(max(ratios),abs=1e-6)


def test_ragged_sixty_way_plus_boolean_ignores_only_declared_padding():
    logits=torch.full((2,60),-torch.inf);logits[0]=torch.arange(60)/20;logits[1,:2]=torch.tensor([1.,2.])
    result=unpack_candidate_logits(logits,[list(range(60)),[32,33]])
    assert [len(r) for r in result]==[60,2]
    assert torch.equal(result[1],torch.tensor([1.,2.]).log_softmax(0))
    assert compare_span_policies(result,result)['passed']


@pytest.mark.parametrize('fault',['nan','padding_zero','padding_nan','shape','integer','duplicate','short','negative'])
def test_rejects_ambiguous_pooler_outputs(fault):
    x=torch.tensor([[1.,2.,3.],[2.,3.,-torch.inf]]);ids=[[1,2,3],[1,2]]
    if fault=='nan':x[0,0]=torch.nan
    if fault=='padding_zero':x[1,2]=0
    if fault=='padding_nan':x[1,2]=torch.nan
    if fault=='shape':x=x[:1]
    if fault=='integer':x=torch.ones((2,3),dtype=torch.long)
    if fault=='duplicate':ids[0]=[1,1,2]
    if fault=='short':ids[1]=[1]
    if fault=='negative':ids[0]=[-1,2,3]
    with pytest.raises(ValueError):unpack_candidate_logits(x,ids)


def test_six_span_joint_gate_not_weakened():
    a=unpack_candidate_logits(torch.tensor([[.012,0.]]*6),[[1,2]]*6)
    b=unpack_candidate_logits(torch.zeros((6,2)),[[1,2]]*6)
    r=compare_span_policies(a,b)
    assert r['per_span_passed'] and not r['joint_ratio_passed']
