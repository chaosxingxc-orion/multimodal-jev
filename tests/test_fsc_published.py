import pytest
from multimodal_jev.fsc_published import parse_prediction, render_reference, score_predictions


def row(path='a'):
    return {'path': path, 'action': 'activate', 'object': 'lights', 'location': 'none'}


def test_published_pseudo_json_and_literal_none():
    raw = '{"action:" "activate"| "object": "lights"| "location": "none"}'
    assert render_reference(row()) == raw
    assert parse_prediction(raw) == {k: row()[k] for k in ('action', 'object', 'location')}
    assert parse_prediction('{"action":"activate","object":"lights","location":null}') is None
    assert parse_prediction(raw+' trailing') is None


def test_all_slots_required_missing_remains_in_denominator():
    refs = [row('a'), row('b'), row('c')]
    preds = [{'example_id': 'a', 'raw_prediction': render_reference(refs[0])},
             {'example_id': 'b', 'raw_prediction': render_reference({**refs[1], 'location': 'kitchen'})}]
    result = score_predictions(refs, preds)
    assert result['joint_accuracy'] == 1/3
    assert result['component_accuracy']['action'] == 2/3
    assert result['missing'] == 1
    assert score_predictions(refs, [])['joint_accuracy'] == 0


def test_duplicate_or_foreign_predictions_rejected():
    pred = {'example_id': 'a', 'raw_prediction': render_reference(row())}
    with pytest.raises(ValueError, match='duplicate'):
        score_predictions([row()], [pred, pred])
    with pytest.raises(ValueError, match='unknown'):
        score_predictions([row()], [{**pred, 'example_id': 'test-secret'}])
