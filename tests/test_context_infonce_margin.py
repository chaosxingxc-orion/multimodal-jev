import math
from pathlib import Path
import pytest
import torch
from multimodal_jev.context_infonce import loss,validate_objective

M=dict(margin=.1,decision_threshold=.5,gap_weight=1.,anchor_weight=.25)
O=dict(kind='candidate_infonce_probability_margin_v1',temperature=1.,probability_coefficient=0.,probability_margin=M)


def calc(z, real=None, world=1, denominator=None):
    y=torch.zeros_like(z,dtype=torch.bool);y[:,0]=True
    real=torch.ones(len(z),dtype=torch.bool) if real is None else real
    return loss(z,y,torch.ones_like(y),real.to(z.dtype),real,
        global_real_requests=denominator or max(1,int(real.sum())),world_size=world,probability_margin=M)


def test_probability_margin_and_threshold_anchors_have_correct_gradient():
    z=torch.tensor([[.1,.05,-.5]],dtype=torch.float64,requires_grad=True)
    value,metrics=calc(z);base=loss(z,torch.tensor([[True,False,False]]),torch.ones_like(z,dtype=torch.bool),torch.ones(1),torch.ones(1,dtype=torch.bool),global_real_requests=1)[0]
    grad,=torch.autograd.grad(value-base,z)
    assert grad[0,0]<0 and grad[0,1]>0 and grad[0,2]==0
    assert metrics['margin_satisfied_count']==0
    z=torch.tensor([[2.,-2.,-3.]],requires_grad=True)
    _,metrics=calc(z)
    assert metrics['probability_margin_numerator']==metrics['threshold_anchor_numerator']==0


def test_saturated_wrong_prediction_has_nonvanishing_anchor_gradient():
    z=torch.tensor([[-100.,100.]],requires_grad=True)
    v,m=calc(z);v.backward()
    assert torch.isfinite(v) and z.grad[0,0]<-1 and z.grad[0,1]>1
    assert m['threshold_anchor_numerator']>99


def test_common_logit_shift_is_detected_despite_same_infonce():
    _,a=calc(torch.tensor([[2.,-2.]]));_,b=calc(torch.tensor([[12.,8.]]))
    assert a['infonce_numerator']==pytest.approx(b['infonce_numerator'],abs=1e-6)
    assert a['anchored_count']==1 and b['anchored_count']==0
    assert b['probability_margin_numerator']>0


def test_dp4_microbatch_tail_gradient_equals_unsplit_population():
    torch.manual_seed(317)
    z=torch.randn(16,3,dtype=torch.float64,requires_grad=True);real=torch.arange(16)<9
    v,_=calc(z,real);expected,=torch.autograd.grad(v,z)
    zz=z.detach().clone().requires_grad_()
    split=sum(calc(zz[i:i+1],real[i:i+1],world=4,denominator=9)[0]/4 for i in range(16))
    actual,=torch.autograd.grad(split,zz)
    torch.testing.assert_close(actual,expected,atol=1e-14,rtol=0)
    assert torch.equal(actual[9:],torch.zeros_like(actual[9:]))


def test_bf16_and_new_objective_actor_wiring():
    from tools.reports.context_infonce_actor import load
    from tools.reports.context_infonce_trainer import generate
    z=torch.tensor([[-1000.,1000.]],dtype=torch.bfloat16,requires_grad=True)
    v,_=calc(z);v.backward();assert torch.isfinite(z.grad).all() and v.dtype==torch.float32
    actor=load(Path.cwd(),temperature=1.,probability_margin=M)
    y=torch.tensor([[True,False]])
    value,_=actor.loss(z,y,torch.ones_like(y),torch.ones(1),torch.ones(1,dtype=torch.bool),stage='instruction',ranking_coefficient=1.,probability_coefficient=0.,kl_coefficient=0.,global_real_requests=1,world_size=1,actions=None,old_joint=None,advantages=None,reference_policy=None)
    assert torch.equal(value,v)
    assert "probability_margin=c['instruction_objective'].get('probability_margin')" in generate(Path.cwd())
    validate_objective(O)


@pytest.mark.parametrize('field,value',[('margin',0),('margin',1),('decision_threshold',.99),('gap_weight',0),('anchor_weight',float('nan'))])
def test_invalid_margin_rejected(field,value):
    with pytest.raises(ValueError):validate_objective(dict(O,probability_margin=dict(M,**{field:value})))
