import pytest
from multimodal_jev.decision_questions import Concept,predicate,compile_menu,predicate_targets,exclusive_class_targets


def test_operator_transfers_by_definition_without_dataset_or_query_branch():
    a=predicate('has_intent','intent_expressed',concept=Concept('calendar','Request creation of a calendar event.'))
    b=predicate('has_intent','intent_expressed',concept=Concept('command','Request activation of a device.'))
    assert a.schema.question.split('\nDefinitions:')[0]==b.schema.question.split('\nDefinitions:')[0]
    assert a.values==b.values==(False,True)
    assert a.schema.question!=b.schema.question


def test_slot_presence_and_intent_binding_have_distinct_semantics():
    slot=Concept('date','A calendar date serving as a requested event date.')
    intent=Concept('create','Request creation of a calendar event.')
    presence=predicate('slot','slot_type_present',concept=slot)
    bound=predicate('bound','intent_slot_bound',intent=intent,slot=slot)
    assert presence.schema.question!=bound.schema.question
    assert 'Co-occurrence' in bound.schema.question
    assert 'Do not extract its value' in presence.schema.question


def test_single_intent_gold_does_not_define_all_semantic_intentions():
    concept=Concept('task','Request cancellation of an item.')
    exclusive=predicate('q','intent_label_member',concept=concept)
    semantic=predicate('q','intent_expressed',concept=concept)
    assert exclusive.schema.question!=semantic.schema.question
    assert 'single-intent' in exclusive.schema.question


def test_unknown_is_explicit_not_missing_label_conversion():
    q=predicate('q','slot_type_present',concept=Concept('date','Calendar date.'),allow_unknown=True)
    assert q.values==(False,True,None)
    assert not hasattr(q,'target')


@pytest.mark.parametrize('kwargs', [dict(operator='guess'),
    dict(operator='slot_type_present'),dict(operator='intent_slot_bound',concept=Concept('x','Meaning.')),
    dict(operator='intent_expressed',concept=Concept('x','Meaning.'),slot=Concept('y','Role.'))])
def test_reject_undefined_or_inapplicable_arguments(kwargs):
    with pytest.raises(ValueError):predicate('q',**kwargs)


def test_no_query_or_gold_argument_can_route_question():
    with pytest.raises(TypeError):
        predicate('q','intent_expressed',concept=Concept('x','Meaning.'),target=True)


def test_changing_gold_cannot_change_fixed_questions():
    menu=compile_menu('intent_label_member',[Concept('a','First class.'),Concept('b','Second class.')])
    before=menu.questions
    a=exclusive_class_targets(menu,'a',label_basis='admitted_exclusive_fixture')
    b=exclusive_class_targets(menu,'b',label_basis='admitted_exclusive_fixture')
    assert a['targets']==[1,0] and b['targets']==[0,1]
    assert menu.questions==before
    assert a['question_ids']==b['question_ids']


def test_partial_slot_annotation_never_creates_false_or_unknown_supervision():
    menu=compile_menu('slot_type_present',[Concept('a','First role.'),Concept('b','Second role.'),Concept('c','Third role.')],allow_unknown=True)
    result=predicate_targets(menu,positives={'a'},negatives={'b'},unavailable={'c'},label_basis='partial_fixture')
    assert result['targets']==[1,0,0] and result['supervised']==[True,True,False]
    assert result['status']==['positive','negative','unavailable']
    with pytest.raises(ValueError):
        predicate_targets(menu,positives={'a'},negatives={'b'},unavailable=set(),label_basis='partial_fixture')


@pytest.mark.parametrize('op',['intent_expressed','slot_type_present'])
def test_single_intent_label_cannot_manufacture_other_task_negatives(op):
    menu=compile_menu(op,[Concept('a','First meaning.'),Concept('b','Second meaning.')])
    with pytest.raises(ValueError,match='semantic absence'):
        exclusive_class_targets(menu,'a',label_basis='single_intent')


def test_conflicting_status_and_unknown_id_are_rejected():
    menu=compile_menu('slot_type_present',[Concept('a','Role.')])
    for pos,neg,missing in [({'a'},{'a'},set()),({'wrong'},set(),set())]:
        with pytest.raises(ValueError):
            predicate_targets(menu,positives=pos,negatives=neg,unavailable=missing,label_basis='fixture')
