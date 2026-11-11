import pytest
from multimodal_jev.temporal_diagnostic import score


def test_invalid_boolean_and_partial_bundle_denominators():
    targets=[dict(example_id='a::'+k,target=v) for k,v in [('date',True),('time',False),('timeofday',True)]]
    targets.append(dict(example_id='b::date',target=False))
    predictions=[dict(example_id=r['example_id'],raw_text=t) for r,t in zip(targets,
        ['{"value":true}','{"value":false}','{"value":1}','not JSON'])]
    metrics=score(targets,predictions)
    assert metrics['decisions']['correct']==2
    assert metrics['decisions']['valid_boolean']==2
    assert metrics['complete_bundles']==dict(total=1,correct=0)
    assert metrics['partial_bundles']==1
    assert metrics['by_task']['timeofday']['positive_correct']==0
    with pytest.raises(ValueError,match='coverage'):score(targets,predictions[:-1])
    with pytest.raises(ValueError,match='coverage'):score(targets,predictions+[predictions[0]])


def test_all_true_control_and_strict_invalid_formats():
    targets=[dict(example_id=f'{i}::{k}',target=bool(i))
             for i in (0,1) for k in ('date','time','timeofday')]
    predictions=[dict(example_id=r['example_id'],raw_text='{"value":true}') for r in targets]
    metrics=score(targets,predictions)
    assert metrics['decisions']['correct']==3
    assert metrics['complete_bundles']==dict(total=2,correct=1)
    for text in ('{"value":null}','{"value":"false"}',
                 '{"value":true,"value":false}', '```json\n{"value":true}\n```'):
        predictions=[dict(example_id=r['example_id'],raw_text=text) for r in targets]
        assert score(targets,predictions)['decisions']['valid_boolean']==0
