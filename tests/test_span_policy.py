import pytest
torch=pytest.importorskip('torch')
from multimodal_jev.span_policy import sample_spans,joint_logprobs


def test_complete_tuple_probability_matches_exhaustive_product_and_has_gradients():
    from itertools import product
    logits=[torch.tensor([[1.,-1.]],requires_grad=True),torch.tensor([[.2,.3,.4]],requires_grad=True)]
    logp=[x.log_softmax(-1) for x in logits]
    actions=torch.tensor([list(product(range(2),range(3)))])
    joint=joint_logprobs(logp,actions)
    torch.testing.assert_close(joint.exp().sum(-1),torch.ones(1))
    for k,(a,b) in enumerate(actions[0]):
        torch.testing.assert_close(joint[0,k].exp(),logp[0][0,a].exp()*logp[1][0,b].exp())
    (-joint[0,0]).backward()
    assert all(x.grad.abs().sum()>0 for x in logits)


def test_sampling_returns_complete_actions_and_matching_joint_probabilities():
    logp=[torch.tensor([[.1,.9],[.7,.3]]).log(),torch.tensor([[.2,.3,.5],[.1,.6,.3]]).log()]
    actions,old=sample_spans(logp,8,generator=torch.Generator().manual_seed(17))
    assert actions.shape==(2,8,2)
    torch.testing.assert_close(old,joint_logprobs(logp,actions))
    with pytest.raises(ValueError,match='normalized'):sample_spans([logp[0]+1],2)
    with pytest.raises(ValueError,match='support'):joint_logprobs(logp,torch.full((2,8,2),99))


def test_final_only_reward_has_zero_expected_score_gradient_for_auxiliary_action():
    from itertools import product
    auxiliary=torch.tensor([[.3,-.2]],requires_grad=True)
    final=torch.tensor([[-.1,.4]],requires_grad=True)
    logp=[auxiliary.log_softmax(-1),final.log_softmax(-1)]
    actions=torch.tensor([list(product(range(2),range(2)))])
    joint=joint_logprobs(logp,actions)
    reward=(actions[...,1]==1).float()
    # Exact REINFORCE expectation, with sample probabilities treated as weights.
    surrogate=-(joint.exp().detach()*reward*joint).sum()
    aux_grad,final_grad=torch.autograd.grad(surrogate,[auxiliary,final])
    torch.testing.assert_close(aux_grad,torch.zeros_like(aux_grad),atol=1e-7,rtol=0)
    assert final_grad.abs().sum()>0
