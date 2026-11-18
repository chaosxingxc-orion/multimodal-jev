import pytest
import torch
from tools.reports.analyze_context_probability_objectives import components


def test_nll_and_kl_gradients_match_closed_forms():
    p=torch.tensor([.08,.72,.16],dtype=torch.float64)
    q=torch.tensor([.21,.59,.22],dtype=torch.float64)
    values, gradients=components(p,q,1)
    y=torch.tensor([0.,1.,0.],dtype=torch.float64)
    torch.testing.assert_close(gradients['binary_nll'],(p-y)/3)
    torch.testing.assert_close(gradients['reference_kl_sum'],p*(1-p)*(torch.logit(p)-torch.logit(q)))
    assert values['binary_nll']==pytest.approx(values['positive_contribution']+values['negative_contribution'])
    assert values['reference_kl_sum']>0


def test_ranking_gradient_finite_difference_and_shift_invariance():
    p=torch.tensor([.21,.48,.86],dtype=torch.float64);z=torch.logit(p)
    _, gradients=components(p,p,1)
    for j in range(3):
        dz=torch.zeros(3,dtype=torch.float64);dz[j]=1e-5
        high=components((z+dz).sigmoid(),p,1)[0]['ranking']
        low=components((z-dz).sigmoid(),p,1)[0]['ranking']
        assert gradients['ranking'][j].item()==pytest.approx((high-low)/2e-5,abs=1e-10)
    assert abs(float(gradients['ranking'].sum()))<1e-14


def test_confident_wrong_probability_not_hidden_by_negative_average():
    values,_=components([.001,.999,.01],[.1,.5,.01],0)
    assert values['confident_false_negative']==1 and values['confident_false_positives']==1
    assert values['positive_nll']>6 and values['negative_nll_sum']>6


@pytest.mark.parametrize('p',[[0.,.2],[1.,.3],[float('nan'),.5]])
def test_invalid_probabilities(p):
    with pytest.raises(ValueError):components(p,[.5,.5],0)
