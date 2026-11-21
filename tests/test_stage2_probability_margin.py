import hashlib
from pathlib import Path
import pytest
import torch
from multimodal_jev.objective_plugin import ObjectivePlugin
from multimodal_jev import context_dual_credit as dual
from test_objective_plugin import surrogate
from extensions.stage2_probability_margin_objective import DEFAULTS


def loss_at(values, **overrides):
    z=torch.tensor([values],requires_grad=True)
    identity=dict(policy_id='p',input_id='i',query_id='q',update=0)
    record=dual.prepare(z[0].detach(),z[0].detach().sigmoid(),['a','b','c'],
        torch.tensor([True,False,False]),**identity,algorithm=dual.ALGORITHMS[1])
    path=Path(__file__).resolve().parents[1]/'extensions/stage2_probability_margin_objective.py'
    opts=dict(DEFAULTS,ranking_rl_weight=0.,existence_rl_weight=0.,task_rl_weight=0.,
        ranking_auxiliary_coefficient=0.,probability_nll_sum_coefficient=0.,reference_kl_mean_coefficient=0.,
        probability_margin=.2,probability_margin_coefficient=1.)
    opts.update(overrides)
    plugin=ObjectivePlugin(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),options=opts))
    loss,metrics=plugin.hybrid_loss(z,[record],[identity],torch.full_like(z,.5).detach(),torch.tensor([True]),global_real_requests=1)
    return loss,torch.autograd.grad(loss,z)[0],metrics


def test_pushes_gold_up_and_strongest_wrong_down(surrogate):
    loss,gradient,metrics=loss_at([0.,1.,-1.])
    assert gradient[0,0]<0 and gradient[0,1]>0 and gradient[0,2]==0
    assert metrics['probability_margin_satisfied_count']==0
    assert float(loss.detach())>.2


def test_stops_after_gold_margin_met(surrogate):
    loss,gradient,metrics=loss_at([2.,-2.,-3.])
    assert float(loss.detach())==0 and not gradient.any()
    assert metrics['probability_margin_satisfied_count']==1


def test_disabled_margin_has_no_gradient(surrogate):
    loss,gradient,_=loss_at([0.,1.,-1.],probability_margin_coefficient=0.)
    assert float(loss.detach())==0 and not gradient.any()


def test_invalid_margin_rejected(surrogate):
    with pytest.raises(ValueError,match='less than one'):loss_at([0.,1.,-1.],probability_margin=1.)
