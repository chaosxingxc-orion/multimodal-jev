import pytest
torch = pytest.importorskip('torch')
from multimodal_jev.span_hybrid_objective import loss, sft_coefficient


def batch():
    logits = torch.tensor([[.2, -.8, .5], [.3, .1, -.4], [-.2, .5, .1], [.7, -.3, .1]], requires_grad=True)
    y = torch.tensor([[0], [1], [-99], [2]])
    mask = torch.tensor([[True], [True], [False], [True]])
    w = torch.ones(4, 1)
    rw = torch.tensor([1., 2., 1., 0.])
    return logits, y, mask, w, rw


def test_instruction_gradient_matches_weighted_ce_and_microbatch_accumulation():
    z, y, mask, w, rw = batch()
    p = (z.log_softmax(-1),)
    actual, _ = loss(p, y, mask, w, rw, stage='instruction', global_effective_weight=3.)
    reference = (-p[0][0, 0]-2*p[0][1, 1])/3
    assert torch.allclose(actual, reference)
    expected_grad = torch.autograd.grad(reference, z, retain_graph=True)[0]
    # Four simulated mean-reduced ranks, each with one microbatch. Two have no active weight.
    partial = []
    for i in range(4):
        value, _ = loss((p[0][i:i+1],), y[i:i+1], mask[i:i+1], w[i:i+1], rw[i:i+1],
                        stage='instruction', global_effective_weight=3., world_size=4)
        partial.append(value)
    grad = torch.autograd.grad(sum(partial)/4, z)[0]
    assert torch.allclose(grad, expected_grad, atol=1e-7)
    assert torch.count_nonzero(grad[2:]) == 0


def test_hybrid_uses_native_verl_and_preserves_supervised_gradient_at_low_probability():
    pytest.importorskip('verl')
    z = torch.tensor([[0., -12.]], requires_grad=True)
    p = (z.log_softmax(-1),); old = tuple(v.detach() for v in p)
    y = torch.tensor([[1]]); mask = torch.ones(1, 1, dtype=torch.bool); w = torch.ones(1, 1); rw = torch.ones(1)
    rl, _ = loss(p, y, mask, w, rw, stage='task', global_effective_weight=1., behavior=old, coefficient=0.)
    hybrid, _ = loss(p, y, mask, w, rw, stage='task', global_effective_weight=1., behavior=old, coefficient=.1)
    gr = torch.autograd.grad(rl, z, retain_graph=True)[0]
    gh = torch.autograd.grad(hybrid, z)[0]
    assert gr[0, 1] < 0 and gh[0, 1] < -.099
    assert abs(gh[0, 1]) > abs(gr[0, 1])*1000


def test_decay_is_resumable_and_does_not_restart_or_silently_extrapolate():
    values = [sft_coefficient(i, 100) for i in range(100)]
    assert values[0] == 1. and values[-1] == .1
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert [sft_coefficient(i, 100) for i in range(40, 100)] == values[40:]
    with pytest.raises(ValueError):
        sft_coefficient(100, 100)


def test_bad_masks_weights_and_behavior_are_rejected():
    z, y, mask, w, rw = batch(); p = (z.log_softmax(-1),)
    with pytest.raises(ValueError):
        loss(p, y, mask, w, rw, stage='instruction', global_effective_weight=0.)
    with pytest.raises(ValueError, match='behavior'):
        loss(p, y, mask, w, rw, stage='task', global_effective_weight=3.)
    with pytest.raises(ValueError):
        loss(p, y, torch.ones_like(mask), w, rw, stage='instruction', global_effective_weight=4.)
