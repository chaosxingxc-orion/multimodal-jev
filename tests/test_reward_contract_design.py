from fractions import Fraction as F
from pathlib import Path
import runpy

module=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/jev/audit_reward_contract.py'))
reward=module['fixed_reward']


def test_exact_conditional_and_rloo_expectations():
    r=module['audit']()
    assert r['exact_logit_gradient']==r['conditional_gradient']==r['rloo_G2_expected_gradient']
    assert r['final_only_process_gradient']=='0'


def test_missing_family_cannot_transfer_its_budget():
    assert reward(False,{'I':[1],'S':[]},{'I':[True],'S':[]},{'I':(1,F(1,10)),'S':(0,F(1,10))})==F(1,10)


def test_changing_masked_target_cannot_change_reward():
    masks={'I':[True,False]};budgets={'I':(2,F(1,10))}
    assert reward(1,{'I':[1,0]},masks,budgets)==reward(1,{'I':[1,1]},masks,budgets)==F(21,20)


def test_duplicating_configured_questions_does_not_double_reward():
    assert reward(0,{'I':[1]}, {'I':[True]}, {'I':(1,F(1,10))})==reward(0,{'I':[1,1]}, {'I':[True,True]}, {'I':(2,F(1,10))})
