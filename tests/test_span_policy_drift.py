import math
import pytest
from multimodal_jev.span_policy_drift import diagnose


def lp(p):return list(map(math.log,p))


def test_on_policy_zero_loss_does_not_mean_no_gradient():
    d=diagnose(lp([.1,.2,.7]),lp([.1,.2,.7]),0)
    for k in ('kl_old_to_new','total_variation','old_mass_clipped_by_advantage','postupdate_clipped_surrogate'):
        assert d[k]==pytest.approx(0,abs=1e-15)


def test_exact_unclipped_gain_and_clipped_mass():
    d=diagnose(lp([.1,.9]),lp([.3,.7]),0)
    assert d['gold_probability_delta']==pytest.approx(.2)
    assert d['postupdate_unclipped_surrogate']==pytest.approx(.2)
    assert d['old_mass_outside_clip']==pytest.approx(1.)
    assert d['old_mass_clipped_by_advantage']==pytest.approx(1.)
    assert d['postupdate_clipped_surrogate']<d['gold_probability_delta']


def test_detrimental_move_outside_ratio_is_not_advantage_clipped():
    d=diagnose(lp([.3,.7]),lp([.1,.9]),0)
    assert d['old_mass_outside_clip']==pytest.approx(1.)
    assert d['old_mass_clipped_by_advantage']==0
    assert d['postupdate_clipped_surrogate']==pytest.approx(-.2)


@pytest.mark.parametrize('new',[[0.,0.],[float('nan'),0.],[-1.]])
def test_rejects_incomplete_or_invalid_support(new):
    with pytest.raises(ValueError):diagnose(lp([.5,.5]),new,0)
