from pathlib import Path
import runpy
import pytest

resolve=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/jev/audit_slurp_intent_namespace.py'))['resolve']


def test_shared_action_alias_cannot_choose_category_without_scenario():
    pairs={'calendar_query':['calendar','query'],'weather_query':['weather','query']}
    assert resolve({'intent':'query','scenario':'calendar','action':'query'},pairs)=='calendar_query'
    assert resolve({'intent':'query','scenario':'weather','action':'query'},pairs)=='weather_query'


def test_raw_intent_cannot_override_canonical_pair():
    assert resolve({'intent':'untrusted_name','scenario':'calendar','action':'query'},
                   {'calendar_query':['calendar','query']})=='calendar_query'


def test_unknown_and_ambiguous_pairs_rejected():
    row={'scenario':'calendar','action':'query'}
    with pytest.raises(ValueError,match='outside'):resolve(row,{'weather_query':['weather','query']})
    with pytest.raises(ValueError,match='non-injective'):
        resolve(row,{'a':['calendar','query'],'b':['calendar','query']})
