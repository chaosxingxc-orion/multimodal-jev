from copy import deepcopy
import pytest
from test_span_rule_balance import fixture
from multimodal_jev.span_rule_metrics import score_rules


def test_gold_fixture_recovers_every_metric_and_complement():
    rows=fixture();pred=[dict(id=r['id'],predictions=r['targets']) for r in rows]
    result=score_rules(rows,pred)
    assert all(r['final_accuracy']==r['balanced_accuracy']==r['all_span_accuracy']==1 for r in result['per_rule'])
    assert all(r['executor_accuracy']==1 and r['rule_consistency']==1 for r in result['per_rule'] if r['arm']!='final_only')
    assert all(r['n']==r['prediction_flips']==r['both_correct'] for r in result['complement_response'].values())


def test_constant_true_is_chance_balanced_and_fails_complement():
    rows=fixture();pred=[dict(id=r['id'],predictions=[True]*len(r['targets'])) for r in rows]
    result=score_rules(rows,pred)
    assert all(r['balanced_accuracy']==0.5 for r in result['per_rule'])
    assert all(r['prediction_flips']==r['both_correct']==0 for r in result['complement_response'].values())


def test_predicted_fact_executor_does_not_substitute_gold():
    rows=fixture();pred=[dict(id=r['id'],predictions=list(r['targets'])) for r in rows]
    for r in pred:
        if len(r['predictions'])==3:r['predictions'][:2]=[False,False]
    result=score_rules(rows,pred)
    assert all(r['final_accuracy']==1 for r in result['per_rule'])
    assert any(r['executor_accuracy']<1 for r in result['per_rule'] if r['arm']!='final_only')


def test_missing_or_duplicate_predictions_rejected():
    rows=fixture();pred=[dict(id=r['id'],predictions=r['targets']) for r in rows]
    for bad in (pred[:-1],pred+[deepcopy(pred[0])]):
        with pytest.raises(ValueError):score_rules(rows,bad)
