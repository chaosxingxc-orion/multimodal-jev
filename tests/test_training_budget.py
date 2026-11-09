import pytest
from multimodal_jev.training_budget import batch_budget,sum_budgets


def test_uneven_effective_lengths_and_reference_cost():
    result=batch_budget([10,20],[2,5],[92,185],gradient_passes=1,no_grad_passes=2,optimizer_steps=1)
    assert result==dict(sequences=2,prompt_tokens=30,response_tokens=7,audio_frames=277,
                       gradient_sequences=2,gradient_tokens=37,no_grad_sequences=4,
                       no_grad_tokens=74,optimizer_steps=1)
    assert sum_budgets([result,result])=={key:2*value for key,value in result.items()}


def test_zero_advantage_still_accounts_for_parity_work():
    budget=batch_budget([10],[3],[92],gradient_passes=0,no_grad_passes=1,optimizer_steps=0)
    assert budget['no_grad_tokens']==13 and budget['gradient_tokens']==0
    assert budget['optimizer_steps']==0 and budget['response_tokens']==3


@pytest.mark.parametrize('prompts,responses,audio', [([],[],[]),([2],[3,4],[5]),([True],[2],[3]),([2],[0],[3])])
def test_invalid_lengths_are_rejected(prompts,responses,audio):
    with pytest.raises(ValueError):
        batch_budget(prompts,responses,audio,gradient_passes=1,no_grad_passes=0,optimizer_steps=1)


def test_missing_accounting_is_not_silently_zero():
    with pytest.raises(ValueError,match='fields'):sum_budgets([{'optimizer_steps':1}])
