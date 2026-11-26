from collections import Counter
from unittest.mock import patch
import pytest
from multimodal_jev.card_stratified_sampling import sample
from tools.reports.experiment_package import workflow


def test_fixed_replay_and_source_diversity():
    kinds = ['error','tie','replay','boundary','boundary','boundary','boundary']
    rows = [dict(stratum=k,group=g,gold=0,competitor=1,synthetic=False)
            for k,g in zip(kinds, ['a','b','a','a','b','c','d'])]
    policy = dict(retain_all=['error','tie'], replay_recordings=1, budget=5, seed=317, many_rivals=5)
    picked = sample(rows,[0]*len(rows),policy)
    assert picked == [0,1,2,5,6]
    assert len(sample(rows,[0]*len(rows),dict(policy,budget=7))) == 7
    with pytest.raises(ValueError):
        sample(rows,[0]*len(rows),dict(policy,budget=2))


def test_through_never_calls_later_steps():
    spec = dict(stages={'data':[dict(id='pool')],'train':[dict(id='view'),dict(id='actor')],
                        'evaluate':[dict(id='eval')]})
    with patch('tools.reports.experiment_package.step_result',
               side_effect=lambda s,*a:dict(id=s['id'],status='audited',evidence=[])) as call:
        results = workflow(spec,{},None,None,through='view')
    assert [x.args[0]['id'] for x in call.call_args_list] == ['pool','view']
    assert [r['status'] for r in results[2:]] == ['outside_requested_scope']*2
