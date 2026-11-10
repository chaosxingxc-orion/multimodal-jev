import pytest
import torch

from multimodal_jev.omni_cpu_expert_projection import fp32_grouped_linear


@pytest.mark.parametrize('transposed', [False, True])
def test_grouped_output_and_gradients_against_per_token_equations(transposed):
    torch.manual_seed(19)
    x = torch.randn(5, 7, requires_grad=True)
    w = torch.randn(4, 3, 7)
    if transposed:
        w = w.transpose(-2, -1).contiguous()
    w.requires_grad_()
    bias = torch.randn(5, 3, requires_grad=True)
    offsets = torch.tensor([2, 2, 5, 5], dtype=torch.int32)
    result = fp32_grouped_linear(x, w, offsets, bias, transposed)
    # Independent token-to-expert mapping includes empty experts.
    expected = torch.stack([x[t] @ (w[e] if transposed else w[e].t())
                            for t, e in enumerate([0, 0, 2, 2, 2])]) + bias
    scale = torch.randn_like(result)
    actual_grads = torch.autograd.grad((result * scale).sum(), (x, w, bias), retain_graph=True)
    expected_grads = torch.autograd.grad((expected * scale).sum(), (x, w, bias))
    torch.testing.assert_close(result, expected)
    for actual, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, reference)
    assert torch.count_nonzero(actual_grads[1][[1, 3]]) == 0


def test_bf16_ambient_autocast_cannot_demote_projection(monkeypatch):
    torch.manual_seed(21)
    x = torch.randn(5, 17).bfloat16().requires_grad_()
    w = torch.randn(2, 11, 17).bfloat16().requires_grad_()
    calls = []
    original = torch.mm

    def checked(a, b):
        calls.append((a.dtype, b.dtype, torch.is_autocast_enabled('cpu')))
        return original(a, b)

    monkeypatch.setattr(torch, 'mm', checked)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = fp32_grouped_linear(x, w, torch.tensor([2, 5]))
    assert calls == [(torch.float32, torch.float32, False)] * 2
    reference = torch.cat([original(x[:2].float(), w[0].float().t()),
                           original(x[2:].float(), w[1].float().t())]).bfloat16()
    assert torch.equal(actual, reference)
    actual.float().square().sum().backward()
    assert x.grad is not None and w.grad is not None
    assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()


@pytest.mark.parametrize('offsets', [[2, 4], [4, 3], [-1, 5]])
def test_rejects_missing_or_invalid_routes(offsets):
    with pytest.raises(ValueError, match='offsets'):
        fp32_grouped_linear(torch.ones(5, 7), torch.ones(2, 3, 7), torch.tensor(offsets))
