import pytest
from multimodal_jev.context_event_metrics import ContextEventMetrics


def test_set_recovery_and_top1_are_distinct():
    m=ContextEventMetrics(['a','b'])
    m.add('q1','a',[.8,.7]);m.add('q2','b',[.1,.2])
    r=m.result()
    assert r['query_macro_set_f1']==pytest.approx(1/3)
    assert r['exact_set']==0 and r['top1_event_accuracy']==1 and r['top1_macro_f1']==1
    assert r['mean_selected_count']==1 and r['set_precision_defined_queries']==1
    assert r['candidate_brier']==pytest.approx((.04+.49+.01+.64)/4)


def test_fixed_order_ties_absent_classes_and_duplicate_rejection():
    m=ContextEventMetrics(['a','b','c']);m.add('q','a',[.5,.5,.1])
    r=m.result();assert r['top1_event_accuracy']==1 and r['top1_macro_f1']==pytest.approx(1/3)
    with pytest.raises(ValueError):m.add('q','a',[.8,.1,.1])
    with pytest.raises(ValueError):m.add('bad','b',[float('nan'),.1,.1])
    assert m.result()==r
