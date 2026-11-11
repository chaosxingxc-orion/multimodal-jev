import itertools

import pytest

from multimodal_jev.decision_rules import Fact, contract, evaluate, family, question
from multimodal_jev.thinking import ThinkingSpec

D={'a':'The request asks to create a calendar event.', 'b':'The request contains a person in the defined participant role.'}
A={'op':'fact','key':'a'};B={'op':'fact','key':'b'}


def test_all_boolean_compositions_against_independent_truth_tables():
    for a,b in itertools.product((False,True),repeat=2):
        facts={'a':Fact(a,'reviewed'),'b':Fact(b,'reviewed')}
        rules=[({'op':'all','args':[A,B]},a and b),
               ({'op':'any','args':[A,B]},a or b),
               ({'op':'except','base':A,'exception':B},a and not b),
               ({'op':'not','arg':{'op':'all','args':[A,B]}},not (a and b))]
        for rule,expected in rules:
            assert evaluate(rule,D,facts)['value'] is expected


def test_missing_and_conflicting_supervision_is_not_false_or_abstention_gold():
    for status in ('unavailable','conflict'):
        facts={'a':Fact(True,'reviewed'),'b':Fact(None,status)}
        result=evaluate({'op':'all','args':[A,B]},D,facts)
        assert result['value'] is None and result['supervision_status']=='unavailable'
        assert result['actor_abstention_label'] is False
        assert result['unresolved_fact_keys']==['b']
        assert evaluate({'op':'any','args':[A,B]},D,facts)['value'] is True
    assert evaluate({'op':'not','arg':B},D,{})['value'] is None
    assert evaluate({'op':'all','args':[A,B]},D,{'a':Fact(False,'reviewed')})['value'] is False


def test_requirements_are_separate_from_rule_truth_and_rule_change_recomputes_gold():
    thinking=ThinkingSpec('roles-v1',('Check the role in the complete request.',))
    schema,actual=contract({'op':'except','base':A,'exception':B},D,thinking)
    assert actual==thinking and schema.allow_unknown is False
    assert D['a'] in schema.question and D['b'] in schema.question
    assert schema.id=='configured-rule'
    facts={'a':Fact(True,'reviewed'),'b':Fact(True,'reviewed')}
    assert evaluate({'op':'all','args':[A,B]},D,facts)['value'] is True
    assert evaluate({'op':'except','base':A,'exception':B},D,facts)['value'] is False
    with pytest.raises(ValueError):contract(A,D,None)


def test_family_ignores_fact_names_and_commutative_order():
    assert family({'op':'all','args':[A,{'op':'not','arg':B}]},D)==family({'op':'all','args':[{'op':'not','arg':A},B]},D)
    assert family({'op':'all','args':[A,B]},D)!=family({'op':'any','args':[A,B]},D)
    assert question({'op':'not','arg':A},D)!=question(A,D)


@pytest.mark.parametrize('rule', [
    {'op':'fact','key':'missing'}, {'op':'fact','key':'a','gold':True},
    {'op':'all','args':[]}, {'op':'any','args':[A]}, {'op':'not','args':[A]},
    {'op':'except','base':A}, {'op':'python','code':'True'},
])
def test_reject_ambiguous_or_unregistered_rules(rule):
    with pytest.raises(ValueError):evaluate(rule,D,{})


@pytest.mark.parametrize('value,status', [(True,'unavailable'),(False,'conflict'),(None,'reviewed'),(1,'reviewed'),(True,'automatic')])
def test_require_explicit_review(value,status):
    with pytest.raises(ValueError):Fact(value,status)
