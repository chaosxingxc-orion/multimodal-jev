from fractions import Fraction as F
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from audit_semantic_menu_budget import coefficients,process_reward
from build_semantic_actor_menu import compile_panels


def weights():
    panels=compile_panels([f'i{k}' for k in range(65)],[f's{k}' for k in range(55)])
    return panels,coefficients(panels,{'I_present':4,'S_present':4},{'I_present':F(1,10),'S_present':F(1,10)})


def test_ragged_tail_not_boosted_and_maximum_budget_preserved():
    panels,ws=weights()
    assert {w for row in ws for w in row.values()}=={F(1,40)}
    assert len(ws[-1])==1 and sum(ws[-1].values())==F(1,40)
    assert max(sum(row.values()) for row in ws)==F(1,5)
    assert sum(sum(row.values()) for row in ws)/17==F(3,17)


def test_missing_labels_cannot_redistribute_reward_or_remove_action_support():
    _,ws=weights();row=ws[0];keys=list(row)
    c={k:True for k in row};m={k:k==keys[0] for k in row}
    assert process_reward(row,c,m)==F(1,40)
    c[keys[-1]]=False
    assert process_reward(row,c,m)==F(1,40)
    assert len(row)==8
    assert process_reward(row,c,{k:False for k in row})==0


def test_duplicate_and_overcapacity_rejected():
    panels,_=weights()
    with pytest.raises(ValueError,match='duplicate'):
        coefficients(panels+panels[:1],{'I_present':4,'S_present':4},{'I_present':F(1,10),'S_present':F(1,10)})
    with pytest.raises(ValueError,match='exceeds'):
        coefficients(panels,{'I_present':3,'S_present':4},{'I_present':F(1,10),'S_present':F(1,10)})


def test_same_action_final_correctness_dominates_any_process_pattern():
    _,ws=weights()
    for row in ws:
        assert F(1)-sum(row.values())>=F(4,5)
