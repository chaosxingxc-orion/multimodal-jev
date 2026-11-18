import pytest
import torch

from tools.reports.diagnose_context_probability_arithmetic import probability_logs


def test_actual_failure_probability_rounding_exceeds_old_log_bound():
    z = torch.tensor([[3.84375]], requires_grad=True)
    # Exact archived vLLM probability, not a fitted tolerance.
    p = torch.tensor([[0.9790356755256653]])
    behavior = torch.stack((torch.log1p(-p), torch.log(p)), -1)
    stable = torch.stack((torch.nn.functional.logsigmoid(-z), torch.nn.functional.logsigmoid(z)), -1)
    assert float((stable-behavior).abs().max()) > 2e-6
    # CPU sigmoid need not equal Ascend sigmoid. This only establishes that
    # the two arithmetic paths can differ with the same BF16-representable z.
    assert abs(float(z.sigmoid())-float(p)) < 2e-7


def test_probability_path_preserves_gradient_and_sampler_distribution():
    z = torch.tensor([[-12., -3., 0., 3., 6.]], requires_grad=True)
    logs = probability_logs(z)
    p = z.detach().sigmoid()
    assert torch.equal(logs.detach(), torch.stack((torch.log1p(-p), torch.log(p)), -1))
    weights = torch.tensor([[[.2, .9], [.8, .1], [.4, .6], [.7, .3], [.1, .9]]])
    gradient, = torch.autograd.grad((logs*weights).sum(), z)
    expected = weights[..., 1]*(1-p)-weights[..., 0]*p
    torch.testing.assert_close(gradient, expected, atol=2e-7, rtol=2e-6)
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize('value', [100., -1000., float('nan'), float('inf')])
def test_no_clamping_or_saturated_policy_admission(value):
    with pytest.raises(ValueError):
        probability_logs(torch.tensor([[value]], requires_grad=True))
