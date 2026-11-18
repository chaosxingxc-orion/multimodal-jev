from pathlib import Path
import runpy

import pytest

api = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tools/jev/audit_intent_definition_mapping.py'))
recording_fit = api['recording_fit']


def test_missing_source_review_does_not_propagate_positive():
    judgments = {('1', 'intent'): {'definition_fit': 'supported'}}
    assert recording_fit(['1', '2'], 'intent', judgments) == 'partial_source_review'
    assert recording_fit(['2'], 'intent', judgments) == 'unassessed'


def test_different_definition_is_not_an_implicit_negative():
    judgments = {('1', 'other'): {'definition_fit': 'supported'}}
    assert recording_fit(['1'], 'intent', judgments) == 'unassessed'


def test_disagreement_and_unresolved_are_not_collapsed_to_negative():
    judgments = {('1', 'intent'): {'definition_fit': 'supported'},
                 ('2', 'intent'): {'definition_fit': 'not_supported'},
                 ('3', 'intent'): {'definition_fit': 'unresolved'}}
    assert recording_fit(['1', '2'], 'intent', judgments) == 'source_fit_disagreement'
    assert recording_fit(['3'], 'intent', judgments) == 'unresolved'
    assert recording_fit(['2'], 'intent', judgments) == 'not_supported'


def test_fit_cannot_authorize_exclusive_truth():
    m = {'scenario': 'a', 'action': 'b', 'sentence': 'example'}
    c = {'semantic_id': 'a_b'}
    d = dict(metadata_sha256=api['object_sha'](m), reference='example', native_intent='a_b',
             semantic_id='a_b', definition_sha256=api['object_sha'](c), definition_fit='supported',
             native_membership=True, proposed_definition_fit=True, disagrees_with_native_membership=False,
             exclusive_I_class_truth=None, optimization_allowed=False)
    api['verify_decision'](d, m, c)
    d['exclusive_I_class_truth'] = True
    with pytest.raises(ValueError, match='exclusive truth'):
        api['verify_decision'](d, m, c)


def test_stale_definition_rejected():
    m = {'scenario': 'a', 'action': 'b', 'sentence': 'example'}
    d = dict(metadata_sha256=api['object_sha'](m), reference='example', native_intent='a_b',
             semantic_id='a_b', definition_sha256='stale')
    with pytest.raises(ValueError, match='definition revision'):
        api['verify_decision'](d, m, {'semantic_id': 'a_b'})
