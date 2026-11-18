import pytest
import torch

from multimodal_jev.span_packed_attention import packed_causal_attention


def dense(q, k, v, scale):
    groups = q.shape[0] // k.shape[0]
    k = k.repeat_interleave(groups, 0)
    v = v.repeat_interleave(groups, 0)
    scores = (q @ k.transpose(-1, -2)) * scale
    mask = torch.ones(scores.shape[-2:], dtype=torch.bool).triu(1)
    return (scores.masked_fill(mask, -torch.inf).softmax(-1) @ v).transpose(0, 1)


def fake_varlen(q, k, v, cq, ck, **kwargs):
    assert torch.equal(cq, ck) and kwargs['causal'] and kwargs['dropout_p'] == 0.
    return torch.cat([dense(q[a:b].transpose(0, 1), k[a:b].transpose(0, 1),
                            v[a:b].transpose(0, 1), kwargs['softmax_scale'])
                      for a, b in zip(cq[:-1], cq[1:])])


def test_values_gradients_causality_and_zero_padding():
    torch.manual_seed(317)
    q = torch.randn(3, 4, 7, 5, dtype=torch.float64, requires_grad=True)
    k = torch.randn(3, 2, 7, 5, dtype=torch.float64, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    lengths = [2, 7, 4]
    calls = []
    def kernel(*args, **kwargs):
        calls.append(1)
        return fake_varlen(*args, **kwargs)
    actual = packed_causal_attention(q, k, v, lengths, scaling=.3, kernel=kernel)
    expected = torch.stack([torch.nn.functional.pad(dense(q[i, :, :n], k[i, :, :n],
                              v[i, :, :n], .3), (0, 0, 0, 0, 0, 7-n))
                            for i, n in enumerate(lengths)])
    assert len(calls) == 1
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    grads = torch.autograd.grad(actual, (q, k, v), upstream, retain_graph=True)
    refs = torch.autograd.grad(expected, (q, k, v), upstream)
    for grad, ref in zip(grads, refs):
        torch.testing.assert_close(grad, ref)
        for i, n in enumerate(lengths):
            assert not grad[i, :, n:].count_nonzero()
    # Changing other requests and future tokens must not affect first output.
    altered = v.detach().clone(); altered[1:] *= 100; altered[0, :, 1:] += 20
    changed = packed_causal_attention(q, k, altered, lengths, scaling=.3, kernel=fake_varlen)
    torch.testing.assert_close(changed[0, 0], actual[0, 0])


@pytest.mark.parametrize('lengths', [[], [0], [4], [True], [1, 2]])
def test_bad_lengths_rejected(lengths):
    q = torch.ones(1, 2, 3, 4)
    with pytest.raises(ValueError):
        packed_causal_attention(q, q, q, lengths, scaling=.5, kernel=fake_varlen)


def test_masked_row_variant_dispatch(monkeypatch):
    import sys
    from types import SimpleNamespace
    from multimodal_jev.span_packed_attention import masked_row_varlen_kernel
    calls = []
    def fusion(q, k, v, heads, layout, **kwargs):
        calls.append((heads, layout, kwargs))
        return (q,)
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace(npu_fusion_attention=fusion))
    q = torch.randn(5, 4, 8, dtype=torch.bfloat16)
    k = torch.randn(5, 2, 8, dtype=torch.bfloat16)
    ends = torch.tensor([0, 2, 5], dtype=torch.int32)
    out = masked_row_varlen_kernel(q, k, k, ends, ends, softmax_scale=.3)
    assert out is q and out.dtype == torch.bfloat16
    heads, layout, options = calls[0]
    assert (heads, layout) == (4, 'TND')
    assert options['actual_seq_qlen'] == options['actual_seq_kvlen'] == (2, 5)
    assert options['sparse_mode'] == 3 and options['inner_precise'] == 2
    assert options['keep_prob'] == 1. and options['next_tockens'] == 0
    mask = options['atten_mask']
    assert mask.shape == (2048, 2048) and mask.dtype == torch.bool
    assert mask[0, 1] and not mask[1, 0] and not mask.diagonal().any()
