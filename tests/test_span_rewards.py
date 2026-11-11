import pytest
torch=pytest.importorskip('torch')
from multimodal_jev.span_rewards import decision_rewards


def test_final_priority_masked_auxiliary_and_no_reward_for_question_count():
    actions=torch.tensor([[[1,0,1],[0,1,0],[0,0,1]]])
    targets=torch.tensor([[1,-100,1]]);mask=torch.tensor([[True,False,True]])
    r=decision_rewards(actions,targets,mask,2,.2)
    torch.testing.assert_close(r['total'],torch.tensor([[1.2,0.,1.]]))
    duplicated=decision_rewards(torch.cat([actions[:,:,:1],actions],-1),
        torch.tensor([[1,1,-100,1]]),torch.tensor([[True,True,False,True]]),3,.2)
    torch.testing.assert_close(r['total'],duplicated['total'])
    noaux=decision_rewards(actions,targets,torch.tensor([[False,False,True]]),2,.2)
    torch.testing.assert_close(noaux['total'],noaux['final'])


def test_missing_final_gold_not_silently_rewarded():
    with pytest.raises(ValueError,match='Final'):
        decision_rewards(torch.zeros(1,2,3,dtype=torch.long),torch.zeros(1,3,dtype=torch.long),
                         torch.tensor([[True,True,False]]),2,.2)
