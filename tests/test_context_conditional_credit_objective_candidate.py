import itertools
import pytest
import torch
from tools.reports.context_conditional_credit_objective_candidate import conditional_credit_loss


def all_actions(n):
    return torch.tensor(list(itertools.product([0, 1], repeat=n)), dtype=torch.float64)


def pairs(actions, reward):
    columns = []
    for j in range(actions.shape[-1]):
        zero = actions.clone(); zero[:, j] = 0
        one = actions.clone(); one[:, j] = 1
        columns.append(torch.stack([reward(zero), reward(one)], dim=-1))
    return torch.stack(columns, dim=1)


@pytest.mark.parametrize('kind', ['singleton', 'coupled', 'additive'])
def test_exhaustive_expected_shared_parameter_gradient(kind):
    # A shared trainable map: compare parameter gradients, not only logits.
    actions = all_actions(3)
    matrix = torch.tensor([[1., .2], [-.7, .8], [.4, -1.]], dtype=torch.float64)
    theta = torch.tensor([.3, -.8], dtype=torch.float64, requires_grad=True)
    p = (matrix @ theta).sigmoid()
    if kind == 'singleton':
        reward = lambda a: 2*a[:, 0]/(a.sum(-1)+1)
    elif kind == 'coupled':
        reward = lambda a: a[:, 0]*a[:, 1] - .3*a[:, 1]*a[:, 2] + .2*a[:, 2]
    else:
        reward = lambda a: a[:, 0] + .125*a[:, 1] - .125*a[:, 2]
    mass = torch.where(actions.bool(), p, 1-p).prod(-1)
    exact, = torch.autograd.grad((mass*reward(actions)).sum(), theta, retain_graph=True)
    rp = pairs(actions, reward)
    # Integrate sample-specific loss gradients under the old distribution.
    loss = sum(mass[i].detach()*conditional_credit_loss(
        p[None], p.detach()[None], rp[i:i+1][None]) for i in range(len(actions)))
    actual, = torch.autograd.grad(-loss, theta)
    torch.testing.assert_close(actual, exact, rtol=1e-12, atol=1e-14)


def test_constant_wrong_samples_have_signal_and_group_duplication_preserves_scale():
    actions = torch.tensor([[0., 1., 0.]]*16, dtype=torch.float64)
    rp = pairs(actions, lambda a: 2*a[:, 0]/(a.sum(-1)+1))[None]
    z = torch.tensor([-3., 3., -3.], dtype=torch.float64, requires_grad=True)
    p = z.sigmoid()[None]
    loss = conditional_credit_loss(p, p.detach(), rp)
    duplicate = conditional_credit_loss(p.repeat(2, 1), p.detach().repeat(2, 1), rp.repeat(2, 2, 1, 1))
    torch.testing.assert_close(loss, duplicate)
    direction, = torch.autograd.grad(-loss, z)
    assert direction[0] > 0 and direction[1] == 0 and direction[2] == 0


def test_additive_process_reward_has_no_direct_final_logit_contribution():
    actions = all_actions(6)
    z = torch.logit(torch.tensor([.17,.73,.29,.81,.43,.67], dtype=torch.float64)).requires_grad_()
    p = z.sigmoid()
    def f1(a, start):
        return 2*a[:, start]/(a[:, start:start+2].sum(-1)+1)
    mass = torch.where(actions.bool(), p, 1-p).prod(-1)
    final = f1(actions, 0)
    mixed = final + .125*f1(actions, 2) + .125*f1(actions, 4)
    gf, = torch.autograd.grad((mass*final).sum(), z, retain_graph=True)
    gm, = torch.autograd.grad((mass*mixed).sum(), z)
    torch.testing.assert_close(gf[:2], gm[:2], rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(gf[2:], torch.zeros_like(gf[2:]), atol=1e-14, rtol=0)
    assert (gm[2:].abs() > 0).all()


@pytest.mark.parametrize('bad', ['stale', 'reward_grad', 'behavior_grad', 'nan', 'boundary', 'shape'])
def test_invalid_contract_rejected(bad):
    p = torch.tensor([[.2, .8]], requires_grad=True)
    old = p.detach().clone()
    rp = torch.zeros(1, 16, 2, 2)
    if bad == 'stale': old[0,0] += .01
    if bad == 'reward_grad': rp.requires_grad_()
    if bad == 'behavior_grad': old.requires_grad_()
    if bad == 'nan': rp[0,0,0,0] = float('nan')
    if bad == 'boundary': p = torch.tensor([[0., .8]]); old = p.clone()
    if bad == 'shape': rp = rp[:,:,:1,:]
    with pytest.raises(ValueError): conditional_credit_loss(p, old, rp)
