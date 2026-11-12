import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('intent_packet',Path(__file__).resolve().parents[1]/'tools/jev/prepare_slurp_intent_review.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def row(uid,scenario,action):
    return dict(request={'id':uid},target={'native_target':dict(scenario=scenario,action=action)})


def test_selection_deterministic_and_distinct_sources():
    groups={'g1':[row('a','email','query'),row('b','email','query')],
            'g2':[row('c','email','query')],'g3':[row('d','play','game')]}
    selected,coverage=module.select_groups(groups,['email_query','play_game'],2,317)
    assert len(selected)==3
    assert coverage['play_game']['shortfall']==1
    assert all(r['review_status']=='pending' and r['optimization_allowed'] is False for r in selected)
    reversed_groups={k:list(reversed(v)) for k,v in reversed(list(groups.items()))}
    assert module.select_groups(reversed_groups,['email_query','play_game'],2,317)==(selected,coverage)


def test_source_label_conflict_rejected():
    with pytest.raises(ValueError,match='source intent conflict'):
        module.select_groups({'g':[row('a','email','query'),row('b','play','game')]},['email_query','play_game'],2,317)


def test_missing_class_reported_not_filled_by_duplicate():
    selected,coverage=module.select_groups({'g':[row('a','email','query')]},['email_query','play_game'],2,317)
    assert len(selected)==1 and coverage['play_game']==dict(available_groups=0,selected_groups=0,shortfall=2)
