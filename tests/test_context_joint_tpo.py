import pytest

torch = pytest.importorskip('torch')

from multimodal_jev.context_joint_tpo import losses, targets
from multimodal_jev.context_choice import distribution


def test_paired_hit_miss_and_detached_anchors():
    n = torch.zeros(2, 4, requires_grad=True)
    c = torch.zeros(2, 3, requires_grad=True)
    idx = torch.tensor([[3, 0], [1, 2]])
    gold = torch.tensor([3, 0])
    spec = dict(rho=.25, tau=1., temperature=1., margin=.1, classification_weight=1.,
                ranking_weight=1., probability_weight=.3, consistency_weight=.1)
    refs = targets(n, c, n, c, idx, gold, spec)
    assert not any(t.requires_grad for t in refs)
    probs = refs[1].exp()
    assert torch.allclose(probs.sum(-1), torch.ones(2))
    before = distribution(c, 'open', 1.).exp()
    assert probs[0, 0] > before[0, 0] and probs[1, -1] > before[1, -1]
    nl, cl = losses(n, c, refs, spec)
    (nl + cl).sum().backward()
    assert n.grad[0, 3] < 0 and n.grad[1, 0] < 0
    assert c.grad[0, -1] < 0 < c.grad[1, -1]
    # A change of slate order must not move credit to a different class
    # (break ties to make the rival identity unique).
    c2 = torch.tensor([[.7, -.3, .4], [.2, -.4, -.1]])
    perm = torch.tensor([1, 0, 2])
    q1 = targets(n,c2,n,c2,idx,gold,spec)
    q2 = targets(n,c2[:,perm],n,c2[:,perm],idx.flip(1),gold,spec)
    assert torch.allclose(q1[0],q2[0]) and torch.allclose(q1[1][:,perm],q2[1])


def test_no_reward_is_geometric_reference_and_has_zero_kl_at_target():
    n = torch.tensor([[1., -1., 2.]])
    c = torch.tensor([[.2, -.7, .4]])
    idx = torch.tensor([[0, 2]])
    spec = dict(rho=.25, tau=1., temperature=1., margin=.1, classification_weight=0.,
                ranking_weight=0., probability_weight=0., consistency_weight=0.)
    qn, qc = targets(n,c,-n,-c,idx,torch.tensor([1]),spec)
    assert torch.allclose(qn, .5 * n)
    expected = .75 * distribution(c, 'open', 1.) + .25 * distribution(-c, 'open', 1.)
    assert torch.allclose(qc, expected.log_softmax(-1))
    coverage = torch.logsumexp(qc[:, :-1], -1) - qc[:, -1]
    actor = torch.cat((qc[:, :-1], coverage[:, None]), -1)
    assert max(t.abs().max().item() for t in losses(qn,actor,(qn,qc),spec)) < 1e-6
