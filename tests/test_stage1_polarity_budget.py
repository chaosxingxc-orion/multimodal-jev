from pathlib import Path
import runpy
import math
import pytest

m=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/jev/audit_stage1_polarity_budget.py'))
w=m['polarity_weights']


def test_two_sided_unclipped_mass_is_half_each():
    p,n=w(20,80,8)
    assert 20*p==80*n==50


def test_rare_positive_is_capped_without_renormalizing_back_up():
    p,n=w(1,9999,8)
    assert p==8 and math.isclose(9999*n,5000)
    assert (p+9999*n)/10000 < 1


def test_unknown_population_not_used_to_upweight_known_labels():
    r=m['profile'](10,10,100,1,1,8)
    assert r['retained_definition_mass']==.2
    assert r['effective_positive_share']==.5


def test_missing_polarity_does_not_transfer_budget():
    assert w(0,100,8)==(0,.5)
    assert w(0,0,8)==(0,0)


def test_streaming_panel_mean_requires_panel_count_multiplier():
    # One audio: final losses3,9; auxiliary losses2,6 on two disjoint panels.
    target=(3+9)/2+.5*(2+6)/2
    per_request=[3+2*.5/2*2,9+2*.5/2*6]
    assert sum(per_request)/2==target
    wrong=[3+.5/2*2,9+.5/2*6]
    assert sum(wrong)/2!=target


@pytest.mark.parametrize('args',[(1,-1,8),(1,2,.5),(float('nan'),1,8)])
def test_invalid_counts_rejected(args):
    with pytest.raises(ValueError):w(*args)
