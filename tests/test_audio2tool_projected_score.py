import copy
import json
from pathlib import Path
import pytest
from multimodal_jev.audio2tool_targets import project
from multimodal_jev.audio2tool_score import load_official,evaluate_projected


@pytest.fixture
def official():
    p=Path(__file__).parents[1]/'.local/research/audio2tool-code-v1/source/audio_benchmark/evaluation/metrics.py'
    if not p.exists():pytest.skip('fixed upstream source unavailable')
    return load_official(p)


def target(i,extras,official):
    return project(dict(id=i,query_idx=i,tool_name='a',extracted_params={'x':1},
        expected_tool_call='a(x=1)',additional_tool_calls=extras),3,official,{'a','b'})


def prediction(t,calls,error=None):return dict(id=t['id'],raw_text=json.dumps(calls),error=error)


def test_unknown_arguments_do_not_reduce_primary_denominator(official):
    a=target(1,['b'],official);b=target(2,['b()'],official)
    calls=[a['primary'],dict(name='b',arguments={'guessed':9})]
    report,details=evaluate_projected([a,b],[prediction(a,calls),prediction(b,b['full_calls'],'timeout')],official)
    assert report['all']['records']==2
    assert report['all']['full_record_metrics']['strict_primary_exact']==.5
    assert report['all']['full_arguments_subset']==dict(eligible_records=1,unavailable_records=1,strict_full_calls_exact=0.)
    assert report['all']['full_population_strict_calls_exact'] is None
    assert details[0]['strict_full_calls_exact'] is None


def test_name_order_and_repeated_calls_separate_from_set_metrics(official):
    t=target(1,['b','b'],official)
    report,_=evaluate_projected([t],[prediction(t,[t['primary'],dict(name='b',arguments={})])],official)
    m=report['all']['full_record_metrics']
    assert m['official_tool_set_exact']==1 and m['strict_ordered_names_exact']==0


@pytest.mark.parametrize('fault',['empty_unknown','fake_full','changed_raw','missing_prediction'])
def test_incomplete_or_repaired_evidence_rejected(official,fault):
    t=target(1,['b'],official)
    if fault=='empty_unknown':t['additional_annotations'][0]['arguments']={}
    if fault=='fake_full':t['full_calls']=[t['primary']]
    if fault=='changed_raw':t['additional_annotations'][0]['raw']='a'
    preds=[] if fault=='missing_prediction' else [prediction(t,[])]
    with pytest.raises(ValueError):evaluate_projected([t],preds,official)


def test_complete_oracle_is_only_a_fixture(official):
    t=target(1,['b()'],official)
    r,_=evaluate_projected([t],[prediction(t,t['full_calls'])],official)
    assert r['all']['full_population_strict_calls_exact']==1
