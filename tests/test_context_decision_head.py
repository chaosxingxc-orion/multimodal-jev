import copy
import io

import pytest
import torch

from multimodal_jev.context_decision_head import ContextDecisionHead


def fixture(dtype=torch.float64):
    torch.manual_seed(317)
    head = ContextDecisionHead(12, 3, 24).to(dtype)
    hidden = torch.randn(2, 9, 12, dtype=dtype)
    valid = torch.tensor([[False, True, True, True, True, True, True, True, False],
                          [True, True, True, True, True, True, False, False, False]])
    pos = torch.tensor([[3, 7], [2, 5]])
    return head, hidden, pos, valid


def test_prefix_padding_batch_isolation_and_history_sensitivity():
    head, h, pos, valid = fixture()
    baseline = head(h, pos, valid)
    altered = h.clone()
    altered[0, 4:] = torch.randn_like(altered[0, 4:]) * 10
    altered[0, 0] = 100
    altered[1] = torch.randn_like(altered[1]) * 10
    torch.testing.assert_close(head(altered, pos, valid)[0, 0], baseline[0, 0], atol=0, rtol=0)
    padded = h.clone()
    padded[~valid] = 1234
    torch.testing.assert_close(head(padded, pos, valid), baseline, atol=0, rtol=0)
    for i in range(2):
        torch.testing.assert_close(head(h[i:i+1], pos[i:i+1], valid[i:i+1]), baseline[i:i+1])
    past = h.clone()
    past[0, 1] = torch.randn(12, dtype=h.dtype) * 3
    assert (head(past, pos, valid)[0] - baseline[0]).abs().min() > 1e-8


def test_sparse_queries_equal_independent_full_layer_outputs_and_gradients():
    sparse, h, pos, valid = fixture()
    full = copy.deepcopy(sparse)
    h_sparse = h.clone().requires_grad_()
    h_full = h.clone().requires_grad_()
    got = sparse.score_logits(h_sparse, pos, valid)
    # Reference computes all queries using explicit per-request attention algebra,
    # independent of sparse gather/mask construction and torch attention kernels.
    context = full.context_norm(h_full)
    q, k, v = torch.nn.functional.linear(context, full.attention.in_proj_weight,
                                          full.attention.in_proj_bias).chunk(3, -1)
    q, k, v = [x.reshape(2, 9, 3, 4).transpose(1, 2) for x in (q, k, v)]
    scores = q @ k.transpose(-1, -2) / 2
    allowed = torch.ones(9, 9, dtype=torch.bool).tril()[None, None] & valid[:, None, None, :]
    # Padding query 0 of row 0 has an empty prefix; give unused rows a safe key.
    safe = allowed.clone()
    safe[:, :, :, 0] |= ~allowed.any(-1)
    weights = scores.masked_fill(~safe, -torch.inf).softmax(-1)
    attended = (weights @ v).transpose(1, 2).reshape(2, 9, 12)
    state = h_full + full.residual_scale * full.attention.out_proj(attended)
    state = state + full.residual_scale * full.ffn(full.ffn_norm(state))
    all_logits = full.projection(full.output_norm(state)).squeeze(-1)
    expected = all_logits.gather(1, pos)
    torch.testing.assert_close(got, expected, atol=1e-12, rtol=1e-12)
    got.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(h_sparse.grad, h_full.grad, atol=1e-12, rtol=1e-12)
    for (name, p), (_, ref) in zip(sparse.named_parameters(), full.named_parameters()):
        assert p.grad is not None and p.grad.abs().sum() > 0, name
        torch.testing.assert_close(p.grad, ref.grad, atol=1e-12, rtol=1e-12)


def test_final_gradient_reaches_history_but_not_future_or_other_request():
    head, h, pos, valid = fixture()
    h.requires_grad_()
    head(h, pos, valid)[0, 0].backward()
    assert h.grad[0, 1:3].abs().sum() > 0
    assert h.grad[0, 4:].abs().sum() == 0
    assert h.grad[0, 0].abs().sum() == 0
    assert h.grad[1].abs().sum() == 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_probability_roundtrip_and_nonzero_initial_gradients(dtype):
    head, h, pos, valid = fixture(dtype)
    prob = head(h, pos, valid)
    assert prob.dtype == torch.float32 and torch.isfinite(prob).all()
    assert ((prob > 0) & (prob < 1)).all()
    # No menu normalization: each decision independently denotes applicability.
    assert not torch.allclose(prob.sum(-1), torch.ones(2))
    logits = head.score_logits(h, pos, valid)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([[0., 1.], [1., 0.]])).backward()
    for name, p in head.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    stream = io.BytesIO()
    torch.save(head.state_dict(), stream)
    stream.seek(0)
    restored = ContextDecisionHead(12, 3, 24).to(dtype)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    torch.testing.assert_close(restored(h, pos, valid), prob, atol=0, rtol=0)


@pytest.mark.parametrize('fault', ['padding', 'future', 'repeated', 'nan', 'mask'])
def test_invalid_inputs_rejected(fault):
    head, h, pos, valid = fixture()
    if fault == 'padding': pos[0, 0] = 0
    if fault == 'future': pos[0, 1] = 9
    if fault == 'repeated': pos[0, 1] = pos[0, 0]
    if fault == 'nan': h[0, 0, 0] = float('nan')
    if fault == 'mask': valid = valid.long()
    with pytest.raises(ValueError): head(h, pos, valid)
