import pytest
from multimodal_jev.mmar import model_input, native_prompt, score, selected_index


def test_native_order_preserves_padding_and_ignores_supervision():
    row = {'question': 'Which?', 'choices': ['Zebra ', '', 'Apple', 'Apple'],
           'audio_path': 'audio/a.wav', 'answer': 'not a candidate'}
    inp = model_input(row)
    assert 'answer' not in inp
    assert native_prompt(inp, lambda q, o: (q, o)) == (
        'Which?', [(0, 'Zebra '), (1, ''), (2, 'Apple'), (3, 'Apple')])


def test_missing_predictions_keep_full_denominator_and_bad_records_fail():
    rows = [{'id': str(i), 'answer': 'B', 'choices': ['A', 'B']} for i in range(3)]
    pred = {'id': '0', 'probabilities': [0.1, 0.9], 'selected_index': 1}
    result = score(rows, [pred], lambda a, p, c: a == p)
    assert result['publisher_accuracy'] == 1 / 3
    assert result['missing'] == 2 and not result['complete']
    with pytest.raises(ValueError, match='duplicate'):
        score(rows, [pred, pred], lambda *args: True)
    with pytest.raises(ValueError, match='disagrees'):
        score(rows, [{**pred, 'selected_index': 0}], lambda *args: True)
    for probs in ([float('nan'), 1], [-1, 2], [0.1, 0.1], [True, 0], [1]):
        with pytest.raises(ValueError):
            selected_index(probs, 2)
