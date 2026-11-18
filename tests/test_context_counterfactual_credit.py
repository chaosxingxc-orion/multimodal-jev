import itertools
import pytest
import torch
from tools.reports.analyze_context_counterfactual_credit import exact_direction, conditional_direction


@pytest.mark.parametrize('gold',[0,1,2])
def test_matches_exhaustive_expected_reward_and_conditional_gradient(gold):
    p=torch.tensor([.07,.61,.92],dtype=torch.float64)
    actions=torch.tensor(list(itertools.product([0,1],repeat=3)))
    z=torch.logit(p).requires_grad_();q=z.sigmoid()
    weights=torch.where(actions.bool(),q,1-q).prod(-1)
    rewards=2*actions[:,gold].double()/(actions.sum(-1)+1)
    expected=(weights*rewards).sum();gradient,=torch.autograd.grad(expected,z)
    value,exact=exact_direction(p,gold)
    torch.testing.assert_close(exact,gradient,rtol=1e-12,atol=1e-14)
    assert value==pytest.approx(float(expected.detach()),abs=1e-14)
    integrated=(conditional_direction(p,gold,actions)*weights.detach()[:,None]).sum(0)
    torch.testing.assert_close(integrated,exact,rtol=1e-12,atol=1e-14)


def test_identical_wrong_samples_have_signal_but_saturation_remains():
    actions=torch.tensor([[0,1,0]]*16)
    for p in ([.1,.9,.01],[1e-9,.9,.01]):
        signal=conditional_direction(p,0,actions).mean(0)
        assert signal[0]>0 and signal[1]==0 and signal[2]==0
    assert float(signal[0])<1e-9


@pytest.mark.parametrize('p',[[0.,.5],[1.,.5],[float('nan'),.5]])
def test_invalid_probability_rejected(p):
    with pytest.raises(ValueError):exact_direction(p,0)
    with pytest.raises(ValueError):conditional_direction(p,0,torch.tensor([[1,0]]))
