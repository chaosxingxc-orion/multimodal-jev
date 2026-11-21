import torch

from multimodal_jev.context_tpo_rewards import RewardSpec, projection, reward_components


def test_pair_rewards_see_both_members_and_inactive_rows():
    spec = RewardSpec(1,1,1,.69,1,.3,.3,1,.1,.3,.1)
    p = torch.tensor([[.2,.8],[.8,.2],[.2,.8],[.7,.3]])
    g = torch.tensor([0,1,0,0])
    signals = reward_components(p,g,torch.ones(4),torch.tensor([1,0,3,2]),
                                torch.tensor([False,False,True,True]),torch.tensor([True,True,False,False]),spec)
    for name in ('class_utility','rank','existence','margin','contrast','hard_negative'):
        assert signals[name][0,0]>0
        if name!='class_utility':
            assert signals[name][0,1]<0
    assert signals['consistency'][2,0]>0 and signals['consistency'][3,0]<0
    assert not signals['contrast'][2:].any() and not signals['consistency'][:2].any()


def test_only_projection_identifies_absolute_probabilities():
    t=torch.tensor([[.2,-.5,1.8]],requires_grad=True)
    z=(t.detach()+1).requires_grad_()
    projection(z,t).sum().backward()
    torch.testing.assert_close(z.grad,(z.detach().sigmoid()-t.detach().sigmoid())/3)
    assert t.grad is None
    torch.testing.assert_close(projection(t,t),torch.zeros(1))
