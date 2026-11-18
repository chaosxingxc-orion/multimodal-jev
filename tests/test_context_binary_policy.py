import itertools

import pytest
import torch

from multimodal_jev.context_binary_policy import binary_logprobs
from multimodal_jev.span_policy import joint_logprobs, sample_spans


def test_sigmoid_and_joint_enumeration_gradient():
    logits=torch.tensor([[.3,-.8,1.2]],dtype=torch.float64,requires_grad=True)
    policy=binary_logprobs(logits)
    assert torch.allclose(torch.stack([p.exp()[:,1] for p in policy],1),logits.sigmoid())
    actions=torch.tensor([list(itertools.product([0,1],repeat=3))])
    joint=joint_logprobs(policy,actions)
    assert torch.allclose(joint.exp().sum(1),torch.ones(1,dtype=torch.float64))
    # Exact-set fixture reward, not a declared production reward. Compare the
    # full score-function expectation against the analytic event probability.
    gold=torch.tensor([1,0,1]);rewards=(actions==gold).all(-1).double()
    surrogate=-(joint.exp().detach()*rewards*joint).sum()
    score_gradient=torch.autograd.grad(surrogate,logits,retain_graph=True)[0]
    p=logits.sigmoid();expected=-(p[0,0]*(1-p[0,1])*p[0,2])
    direct_gradient=torch.autograd.grad(expected,logits)[0]
    assert torch.allclose(score_gradient,direct_gradient,atol=1e-14,rtol=0)


def test_sampling_replay_and_permutation():
    logits=torch.tensor([[.5,-.7],[1.2,.3]])
    policy=binary_logprobs(logits)
    actions,old=sample_spans(policy,16,generator=torch.Generator().manual_seed(317))
    assert torch.equal(old,joint_logprobs(policy,actions))
    assert torch.allclose(old,joint_logprobs(binary_logprobs(logits.flip(1)),actions.flip(2)))
    # Both true is allowed, unlike a categorical candidate menu.
    assert ((actions==1).all(-1)).any()


def test_extremes_stay_finite_and_bf16_normalizes_in_fp32():
    logits=torch.tensor([[-1000.,1000.]],dtype=torch.bfloat16,requires_grad=True)
    policy=binary_logprobs(logits)
    assert all(p.dtype==torch.float32 and torch.isfinite(p).all() for p in policy)
    assert all(torch.allclose(p.logsumexp(-1),torch.zeros(1)) for p in policy)
    (-policy[0][0,1]-policy[1][0,0]).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().min()>0


@pytest.mark.parametrize('bad',[torch.empty(1,0),torch.ones(3),torch.ones(1,2,dtype=torch.long),
                               torch.tensor([[float('nan')]])])
def test_invalid_logits_rejected(bad):
    with pytest.raises(ValueError):binary_logprobs(bad)
