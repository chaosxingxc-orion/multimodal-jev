from dataclasses import replace
import pytest
from multimodal_jev.span_contract import SpanQuestion, SpanRequest
from multimodal_jev.task_schema import Candidate, DecisionSchema
from multimodal_jev.span_candidate_intervention import VARIANTS, from_input, intervene, canonicalize


def request():
    questions = tuple(SpanQuestion(name, DecisionSchema(name, 'choice', name,
        tuple(Candidate(f'v{i}', f'Meaning {i}') for i in range(n)), allow_unknown=False))
        for name, n in [('scenario', 18), ('action', 25), ('intent', 60)])
    return SpanRequest('/audio.wav', questions, 'intent', ('Listen.',))


@pytest.mark.parametrize('variant', VARIANTS)
def test_full_semantic_roundtrip_and_matched_final(variant):
    original = request()
    changed, receipt = intervene(original, variant)
    direct, direct_receipt = intervene(replace(original, questions=(original.questions[-1],)), variant)
    assert direct.questions[-1] == changed.questions[-1]
    assert direct_receipt['intent'] == receipt['intent']
    assert changed.audio == original.audio and changed.requirements == original.requirements
    assert from_input(changed.model_input()) == changed
    for before, after in zip(original.questions, changed.questions, strict=True):
        assert set(before.schema.candidates) == set(after.schema.candidates)
        assert canonicalize(list(after.values), receipt[before.id]) == list(before.values)
        assert before.schema.question == after.schema.question
    if variant == 'native':
        assert changed == original
    else:
        assert receipt['intent']['moved'] > 0
    if variant.startswith('shift'):
        assert receipt['intent']['moved'] == 60


def test_rejects_targets_and_invalid_intervention():
    value = request().model_input()
    value['target'] = 'v0'
    with pytest.raises(ValueError): from_input(value)
    with pytest.raises(ValueError): intervene(request(), 'best')
    with pytest.raises(ValueError): canonicalize([1, 2], {'presented_to_canonical': [0, 0]})


def test_mapping_does_not_depend_on_audio_or_other_questions():
    a, mapping = intervene(request(), 'permutation317')
    b, other = intervene(replace(request(), audio='/other.wav'), 'permutation317')
    assert a.questions == b.questions and mapping == other
