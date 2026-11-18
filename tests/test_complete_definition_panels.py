from fractions import Fraction
from pathlib import Path
import runpy

import pytest

build=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/jev/design_complete_definition_panels.py'))['complete_panels']


def test_complete_unequal_ontologies_have_no_padding_duplicates():
    orders={'intent':list(range(60)),'slot':list(range(55))}
    panels=build(orders)
    assert len(panels)==15
    for family,values in orders.items():
        flattened=[x for p in panels for x in p['families'][family]['definitions']]
        assert flattened==values
        assert sum(len(p['families'][family]['definitions'])*Fraction(p['families'][family]['per_question_weight']) for p in panels)==1
    assert len(panels[13]['families']['slot']['definitions'])==3
    assert panels[14]['families']['slot']['definitions']==[]
    assert sum(Fraction(p['final_weight']) for p in panels)==1


def test_missing_supervision_does_not_amplify_other_questions():
    p=build({'intent':['a','b','c'],'slot':['s']},2)
    weights=[Fraction(x['families']['intent']['per_question_weight']) for x in p]
    assert weights==[Fraction(1,3)]*2
    # If b is unavailable, a and c retain 1/3 each rather than 1/2.
    assert weights[0]+weights[1]==Fraction(2,3)


@pytest.mark.parametrize('orders,width',[({'intent':['a','a'],'slot':['s']},4),({'intent':[],'slot':['s']},4),({'intent':['a'],'slot':['s']},0)])
def test_invalid_coverage_design_rejected(orders,width):
    with pytest.raises(ValueError):build(orders,width)
