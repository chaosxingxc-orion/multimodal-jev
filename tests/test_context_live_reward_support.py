import pytest
from tools.reports.analyze_context_live_reward_support import decision,summarize


def test_frozen_threshold_empty_and_multiselect():
    assert decision([.5,.7],0)['set_f1']==pytest.approx(2/3)
    assert not decision([.5,.7],0)['exact_set']
    assert decision([.49,.48],0)['top1_correct']
    assert decision([.49,.48],0)['set_f1']==0
    assert decision([.6,.4],0)['exact_set']


def test_probability_quality_penalizes_confident_errors():
    good=decision([.99,.01],0);bad=decision([.01,.99],0)
    assert good['binary_nll']<bad['binary_nll']
    assert good['brier']<bad['brier']
    with pytest.raises(ValueError):decision([0.,1.],0)


def test_empty_stratum_has_no_invented_zero_metrics():
    assert summarize([],'behavior')=={'queries':0}
