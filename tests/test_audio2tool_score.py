import json
from pathlib import Path
import pytest
from multimodal_jev.audio2tool_score import parse_prediction,typed,score_record,evaluate,load_official


@pytest.fixture
def official():
    path=Path(__file__).parents[1]/'.local/research/audio2tool-code-v1/source/audio_benchmark/evaluation/metrics.py'
    if not path.exists():pytest.skip('pinned upstream source cache unavailable')
    return load_official(path)


def call(name='switch',value=1):return dict(name=name,arguments={'value':value})


@pytest.mark.parametrize('raw',['```json\n[]\n```','[{"name":"x","name":"y","arguments":{}}]',
    '[{"name":"x","arguments":{"a":NaN}}]','[{"name":"x","arguments":{"a":1e999}}]',
    '{"name":"x","arguments":{}}','[{"name":"x","arguments":{},"explanation":"x"}]'])
def test_strict_parse_rejects_repairs_and_nonfinite(raw):
    with pytest.raises(ValueError):parse_prediction(raw)


def test_types_and_array_order_preserved():
    assert typed(True)!=typed(1)!=typed(1.)!=typed('1')
    assert typed({'a':1,'b':2})==typed({'b':2,'a':1})
    assert typed([1,2])!=typed([2,1])


def test_order_and_duplicate_calls_distinguish_official_set(official):
    gold=[call('a'),call('b')]
    r=score_record(json.dumps(gold[::-1]),gold,official)
    assert r['official_tool_set_exact'] and not r['strict_ordered_exact']
    r=score_record(json.dumps([call('a')]),[call('a'),call('a')],official)
    assert r['official_tool_set_exact'] and not r['strict_ordered_exact']
    assert r['indexed_slot_fn']==1


def test_official_type_normalization_is_reported_separately(official):
    r=score_record(json.dumps([call(value='1')]),[call(value=1)],official)
    assert r['official_primary_exact'] and not r['strict_ordered_exact']


def test_runtime_errors_and_invalid_empty_targets_never_receive_credit(official):
    targets=[dict(id='a',calls=[call()]),dict(id='b',calls=[call()]),dict(id='c',calls=[])]
    predictions=[dict(id='a',raw_text=json.dumps([call()]),error=None),
                 dict(id='b',raw_text=json.dumps([call()]),error='timeout'),
                 dict(id='c',raw_text='invalid',error=None)]
    report,details=evaluate(targets,predictions,official)
    assert report['records']==3 and report['strict_ordered_exact']==1/3
    assert report['official_primary_exact']==1/3 and report['runtime_errors']==1
    assert report['indexed_slot_micro_f1']==2/3
    assert not details[-1]['official_tool_set_exact']


def test_missing_or_duplicate_results_are_incomplete(official):
    targets=[dict(id='a',calls=[]),dict(id='b',calls=[])]
    pred=dict(id='a',raw_text='[]',error=None)
    for predictions in ([pred],[pred,pred]):
        with pytest.raises(ValueError,match='coverage'):evaluate(targets,predictions,official)
