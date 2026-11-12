from dataclasses import replace
import json
from pathlib import Path
import pytest
from multimodal_jev.semantic_spec import SemanticSpec,compile_definition
from multimodal_jev.span_contract import SpanQuestion
from multimodal_jev.task_schema import DecisionSchema,Candidate


def fixture():
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/tasks/slurp-semantic-spec-candidate-v1.json').read_text())
    spec=SemanticSpec.from_dict(config)
    questions=(SpanQuestion('final',DecisionSchema(id='task',decision_type='choice',question='Which task?',
        candidates=(Candidate('a','Create an item.'),Candidate('b','Inspect existing items.')),allow_unknown=False)),)
    return spec,questions


def test_same_definition_for_every_audio_without_target_or_transcript():
    spec,questions=fixture();definition=compile_definition(questions,'final',spec)
    left=definition.for_audio('/a.wav').model_input();right=definition.for_audio('/b.wav').model_input()
    assert left.pop('audio') != right.pop('audio')
    assert left == right
    assert set(left)=={'questions','final_id','requirements','version'}
    assert definition.questions==questions
    assert json.loads(definition.requirements[0])==json.loads(spec.canonical())


@pytest.mark.parametrize('field',['target','transcript','audio','complexity','source_group','routing'])
def test_reject_sample_fields_in_spec(field):
    spec,_=fixture();raw=json.loads(spec.canonical());raw[field]='leaked sample information'
    with pytest.raises(ValueError,match='task-only'):SemanticSpec.from_dict(raw)


def test_rule_change_changes_input_and_identity_without_changing_candidate_support():
    spec,questions=fixture()
    other=replace(spec,binding_rules=('Treat all requests under the alternate declared policy.',))
    first=compile_definition(questions,'final',spec);second=compile_definition(questions,'final',other)
    assert first.semantic_spec_sha256!=second.semantic_spec_sha256
    assert first.requirements!=second.requirements
    assert first.questions==second.questions
    # This checks construction, not that old gold remains valid under new rules.


def test_rebinding_changes_schema_but_preserves_semantic_rules():
    spec,questions=fixture();q=questions[0]
    rebound=(replace(q,schema=replace(q.schema,candidates=(Candidate('Y','Inspect existing items.'),Candidate('X','Create an item.')))),)
    first=compile_definition(questions,'final',spec);second=compile_definition(rebound,'final',spec)
    assert first.requirements==second.requirements
    assert first.questions[0].values==('a','b') and second.questions[0].values==('Y','X')


def test_optional_process_readouts_are_preserved_without_fabricated_labels():
    spec,questions=fixture()
    process=SpanQuestion('property',DecisionSchema(id='property',decision_type='noul',question='Configured property?',allow_unknown=True))
    request=compile_definition((process,*questions),'final',spec).for_audio('/a.wav')
    assert [q.id for q in request.questions]==['property','final']
    assert request.questions[0].values==(False,True,None)
    assert 'target' not in request.model_input()


def test_reserved_markers_and_unknown_rule_fields_are_rejected():
    spec,_=fixture()
    with pytest.raises(ValueError):replace(spec,objective='[JEV_SPAN_1]')
    with pytest.raises(ValueError):replace(spec,evidence_rules=())
