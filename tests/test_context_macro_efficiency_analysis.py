import copy

import pytest

from test_context_macro_efficiency_summary import fixture
from tools.reports.context_macro_efficiency_summary import summarize
from tools.reports.analyze_context_macro_efficiency import analyze


def setup(tmp_path):
    fixture(tmp_path);s=summarize(tmp_path,start_update=0,stop_update=2,warmup_updates=1)
    a=dict(status='finite_dp2_efficiency_audited',updates=[dict(update=i,real_queries=128,branches={
        'ranking':dict(components=100,zero_signal_components=80,mean_weighted_reward=.9),
        'existence':dict(components=200,zero_signal_components=180,mean_weighted_reward=.8)}) for i in range(2)])
    return s,a


def test_nested_phases_are_not_double_counted_and_counterfactual_is_bounded(tmp_path):
    s,a=setup(tmp_path);r=analyze(s,a);w=r['all_updates'];phases=w['phases']
    assert sum(v['seconds'] for v in phases.values())==22
    assert phases['preparation_outside_rollout']['seconds']==2
    assert phases['rollout_other']['seconds']==2
    assert phases['rank_completion_gap']['seconds']==2
    assert phases['actor']['counterfactual_speedup_if_halved']==22/19
    assert w['four_device_update_window_hours']==88/3600
    assert r['after_warmup']['updates']==1
    assert r['all_update_signal']['ranking']['nonzero_signal_fraction']==.2
    assert r['all_update_signal']['existence']['nonzero_components_per_update_second']==40/22
    assert not r['counterfactuals_are_measured_speedups']


@pytest.mark.parametrize('fault',['audit','step','count','timer','negative_signal','nan'])
def test_invalid_evidence_refused(tmp_path,fault):
    s,a=setup(tmp_path)
    if fault=='audit':a['status']='not_audited'
    if fault=='step':a['updates'][0]['update']=3
    if fault=='count':a['updates'].pop()
    if fault=='timer':s['steps'][0]['reference_seconds']=10.
    if fault=='negative_signal':a['updates'][0]['branches']['ranking']['zero_signal_components']=101
    if fault=='nan':a['updates'][0]['branches']['ranking']['mean_weighted_reward']=float('nan')
    with pytest.raises(ValueError):analyze(s,a)
