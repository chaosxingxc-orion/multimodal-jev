import pytest
import json
from pathlib import Path

from multimodal_jev.slurp_complexity import ontology_strata, surface_flags


def test_joint_ambiguity_is_not_just_repeated_action():
    result = ontology_strata({'a': ['alarm', 'set'], 'b': ['alarm', 'query'],
                              'c': ['calendar', 'set'], 'd': ['weather', 'query']})
    assert result['a']['minimum_components'] == 2
    assert result['d']['minimum_components'] == 1
    assert result['d']['action_competitors'] == 2


def test_reject_aliases_as_distinct_classes():
    with pytest.raises(ValueError):
        ontology_strata({'a': ['alarm', 'set'], 'b': ['alarm', 'set']})


def test_surface_flags_are_independent_review_signals():
    value = surface_flags("Don't cancel it, actually move it after lunch", ['date', 'date'])
    assert value['negation_trigger'] and value['correction_trigger']
    assert value['connective_trigger'] and value['deictic_trigger']
    assert value['repeated_annotated_type']
    assert 'complex' not in value
    # An empty annotation is not certified absence of entities.
    assert surface_flags('event', [])['short_reference']


def test_question_packs_have_disjoint_external_supervision():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/'configs/tasks/slurp-discriminative-question-candidates-v1.json').read_text())
    questions = {q['id']: q for q in config['questions']}
    assert len(questions) == 8
    assert config['status'] == 'prepared_not_admitted_for_training'
    for pack in config['packs'].values():
        assert len(pack) == 2 and len(set(pack)) == 2
        assert set(pack) <= questions.keys()
    for q in questions.values():
        classes = [x for group in q['class_sets'].values() for x in group]
        assert len(classes) == len(set(classes))
        assert q['values'] == list(q['class_sets']) + ['outside_family']
        assert q['supervision'] == 'ontology_projection_not_independent_audio_evidence'
