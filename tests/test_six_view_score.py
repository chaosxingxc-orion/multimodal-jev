import copy
import hashlib
import json
from dataclasses import replace
import pytest
from multimodal_jev.consumption import canonical, request_from_example
from multimodal_jev.task_schema import Candidate, DecisionSchema, JevExample
from multimodal_jev.six_view_score import score_six_views


def fixture():
    candidates = (Candidate('a', 'turn on'), Candidate('b', 'turn off'), Candidate('c', 'turn up'))
    native = DecisionSchema('task', 'choice', 'Choose command', candidates)
    views = [
        ('native', native, 'a'),
        ('permutation', replace(native, candidates=tuple(reversed(candidates))), 'a'),
        ('rebind', replace(native, candidates=(Candidate('a', 'turn up'), Candidate('b', 'turn on'), Candidate('c', 'turn off'))), 'b'),
        ('missing_candidate', replace(native, candidates=candidates[1:]), None),
        ('noul.True', DecisionSchema('binary', 'noul', 'Does the utterance request exactly this command: turn on?'), True),
        ('noul.False', DecisionSchema('binary', 'noul', 'Does the utterance request exactly this command: turn off?'), False),
    ]
    requests, targets, predictions = [], [], []
    for variant, schema, target in views:
        meta = {'variant': variant, 'source_group': 'g', 'audio_sha256': 'same-audio'}
        ex = JevExample('g:' + variant, 'train', '/audio', schema, target, meta)
        request = request_from_example(ex); digest = hashlib.sha256(canonical(request)).hexdigest()
        requests.append(request)
        targets.append({'id': ex.example_id, 'target': target, 'metadata': meta, 'request_sha256': digest})
        predictions.append({'example_id': ex.example_id, 'replicate': 0, 'request_sha256': digest,
                            'raw_text': json.dumps({'value': target})})
    return requests, targets, predictions


def test_gold_fixture_passes_all_six_pairs():
    report = score_six_views(*fixture())
    assert report['requests'] == 6 and report['source_groups'] == 1
    assert report['replicates'][0]['all_six_correct_rate'] == 1
    assert all(p['both_correct_rate'] == 1 for p in report['replicates'][0]['pairs'].values())


@pytest.mark.parametrize('value', ['a', True, False, None])
def test_constant_answer_cannot_pass_discriminating_pairs(value):
    requests, targets, predictions = fixture()
    for prediction in predictions: prediction['raw_text'] = json.dumps({'value': value})
    report = score_six_views(requests, targets, predictions)['replicates'][0]
    assert report['all_six_correct'] == 0 and report['pairs']['noul']['both_correct'] == 0
    assert report['pairs']['rebind']['both_correct'] == 0
    assert report['pairs']['missing_candidate']['both_correct'] == 0
    if value == 'a': assert report['pairs']['permutation']['both_correct'] == 1


def test_malformed_null_and_integer_boolean_are_not_correct():
    requests, targets, predictions = fixture()
    predictions[3]['raw_text'] = 'invalid'
    predictions[4]['raw_text'] = '{"value":1}'
    report = score_six_views(requests, targets, predictions)['replicates'][0]
    assert report['by_variant']['missing_candidate']['correct'] == 0
    assert report['by_variant']['noul.True']['correct'] == 0


@pytest.mark.parametrize('change', ['missing_view', 'missing_prediction', 'duplicate_prediction', 'binding', 'rebind_definition', 'noul_question'])
def test_incomplete_or_semantically_changed_populations_rejected(change):
    requests, targets, predictions = fixture()
    if change == 'missing_view': requests.pop(); predictions.pop()
    if change == 'missing_prediction': predictions.pop()
    if change == 'duplicate_prediction': predictions.append(copy.deepcopy(predictions[0]))
    if change == 'binding': predictions[0]['request_sha256'] = 'wrong'
    if change in ('rebind_definition', 'noul_question'):
        i = 2 if change == 'rebind_definition' else 5
        if i == 2: requests[i]['input']['schema']['candidates'][1]['description'] = 'different command'
        else: requests[i]['input']['schema']['question'] = requests[4]['input']['schema']['question']
        digest = hashlib.sha256(canonical(requests[i])).hexdigest()
        targets[i]['request_sha256'] = predictions[i]['request_sha256'] = digest
    with pytest.raises(ValueError): score_six_views(requests, targets, predictions)
