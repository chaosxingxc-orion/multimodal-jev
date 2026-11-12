import pytest
import json
from pathlib import Path

from multimodal_jev.slurp_complexity import ontology_strata, surface_flags, question_pack_profile


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


def partition_fixture():
    return dict(questions=[
        dict(id='domain', values=['a','b','outside_family'], default='outside_family',
             class_sets={'a':['a1','a2'], 'b':['b1','b2']}),
        dict(id='operation', values=['one','two','outside_family'], default='outside_family',
             class_sets={'one':['a1','b1'], 'two':['a2','b2']})], packs={'joint':['domain','operation']})


def test_partition_distinguishes_joint_information_from_majority_shortcut():
    result = question_pack_profile(partition_fixture(), {'a1':2,'a2':2,'b1':2,'b2':2,'other':92})['joint']
    assert result['joint_only_unique_classes'] == ['a1','a2','b1','b2']
    assert result['single_question_unique_classes'] == []
    assert result['unresolved_family_classes'] == []
    assert result['joint_only_recordings'] == 8
    assert result['constant_answer_accuracy'] == {'domain':.92,'operation':.92}


def test_single_question_label_copy_is_not_joint_reasoning_support():
    config = partition_fixture()
    config['questions'][1]['class_sets'] = {k:[k] for k in ['a1','a2','b1','b2']}
    config['questions'][1]['values'] = ['a1','a2','b1','b2','outside_family']
    result = question_pack_profile(config, dict.fromkeys(['a1','a2','b1','b2','other'], 1))['joint']
    assert result['joint_only_unique_classes'] == []
    assert len(result['single_question_unique_classes']) == 4


@pytest.mark.parametrize('mutation', ['overlap','unknown','duplicate_question','unknown_pack'])
def test_reject_invalid_partition(mutation):
    config = partition_fixture()
    if mutation == 'overlap':
        config['questions'][0]['class_sets']['b'].append('a1')
    elif mutation == 'unknown':
        config['questions'][0]['class_sets']['b'].append('absent')
    elif mutation == 'duplicate_question':
        config['questions'].append(config['questions'][0])
    else:
        config['packs']['joint'][0] = 'missing'
    with pytest.raises(ValueError):
        question_pack_profile(config, dict.fromkeys(['a1','a2','b1','b2','other'], 1))
