from types import SimpleNamespace

import pytest
import torch

from tools.reports.context_segmented_attention_candidate import (
    segmented_attention_forward, segmented_causal_attention,
)
from multimodal_jev import span_batch_rollout_candidate as serial


def oracle(q, k, v, cq, ck, **kwargs):
    """Independent dense masked SDPA for each request, including GQA."""
    assert torch.equal(cq, ck)
    assert kwargs['causal'] and kwargs['dropout_p'] == 0.
    chunks = []
    for a, b in zip(cq.tolist()[:-1], cq.tolist()[1:]):
        query = q[a:b].transpose(0, 1)
        key, value = [x[a:b].transpose(0, 1).repeat_interleave(q.shape[1] // k.shape[1], 0)
                      for x in (k, v)]
        chunks.append(torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True, scale=kwargs['softmax_scale']).transpose(0, 1))
    return torch.cat(chunks)


def fixture():
    torch.manual_seed(317)
    lengths = [1, 7, 3]
    positions = torch.cat([torch.arange(n) for n in lengths])
    q = torch.randn(11, 4, 3, dtype=torch.float64, requires_grad=True)
    k = torch.randn(11, 2, 3, dtype=torch.float64, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    return q, k, v, positions, lengths


def test_single_dispatch_values_gradients_and_no_padding():
    q, k, v, positions, lengths = fixture()
    calls = []
    def kernel(*args, **kwargs):
        calls.append((args[0].shape, args[3].tolist()))
        return oracle(*args, **kwargs)
    actual = segmented_causal_attention(q, k, v, positions, lengths, scaling=.4, kernel=kernel)
    refs = []
    for qs, ks, vs in zip(q.split(lengths), k.split(lengths), v.split(lengths)):
        n = len(qs)
        scores = torch.einsum('thd,shd->hts', qs, ks.repeat_interleave(2, 1)) * .4
        scores = scores.masked_fill(torch.ones(n, n, dtype=torch.bool).triu(1), -torch.inf)
        refs.append(torch.einsum('hts,shd->thd', scores.softmax(-1), vs.repeat_interleave(2, 1)))
    expected = torch.cat(refs)
    assert calls == [(torch.Size([11, 4, 3]), [0, 1, 8, 11])]
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    grads = torch.autograd.grad(actual, (q, k, v), upstream, retain_graph=True)
    reference = torch.autograd.grad(expected, (q, k, v), upstream)
    for a, b in zip(grads, reference):
        torch.testing.assert_close(a, b)


def test_request_isolation_causality_and_permutation():
    q, k, v, positions, lengths = fixture()
    def run(a, b, c, pos=positions, lens=lengths):
        return segmented_causal_attention(a, b, c, pos, lens, scaling=.4, kernel=oracle)
    expected = run(q, k, v)
    changed = [x.detach().clone() for x in (q, k, v)]
    for x in changed:
        x[:1] += 100
        x[8:] *= -50
    torch.testing.assert_close(run(*changed)[1:8], expected[1:8], atol=0, rtol=0)
    changed_v = v.detach().clone(); changed_v[4:8] += 100
    torch.testing.assert_close(run(q, k, changed_v)[1:4], expected[1:4], atol=0, rtol=0)
    order = [2, 0, 1]
    new_lengths = [lengths[i] for i in order]
    new_pos = torch.cat([torch.arange(n) for n in new_lengths])
    reordered = [torch.cat([x.split(lengths)[i] for i in order]) for x in (q, k, v)]
    torch.testing.assert_close(run(*reordered, new_pos, new_lengths),
                               torch.cat([expected.split(lengths)[i] for i in order]))


def test_full_projection_flow_matches_existing_serial_path(monkeypatch):
    torch.manual_seed(317)
    weight = torch.randn(8, 16, dtype=torch.float64)
    calls = []
    def qkv(x):
        calls.append(len(x))
        return x @ weight, None
    module = SimpleNamespace(qkv_proj=qkv, q_size=8, kv_size=4,
        num_heads=4, num_kv_heads=2, head_dim=2,
        q_norm=lambda x: torch.nn.functional.normalize(x, dim=-1),
        k_norm=lambda x: torch.nn.functional.normalize(x, dim=-1),
        rotary_emb=lambda p, q, k: (q * (p[:, None]+1), k * (p[:, None]+1)),
        scaling=.4, o_proj=lambda x: (x * .7, None))
    hidden = torch.randn(10, 8, dtype=torch.float64)
    positions = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4, 5, 6])
    actual = segmented_attention_forward(module, positions, hidden, [3, 7], kernel=oracle)
    assert calls == [10]
    monkeypatch.setattr(serial, 'masked_row_varlen_kernel', oracle)
    expected = torch.cat([serial.packed_attention_forward(module, torch.arange(len(x)), x)
                          for x in hidden.split([3, 7])])
    assert calls == [10, 3, 7]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('positions,lengths', [
    ([0, 1, 2], [3, 1]), ([1, 2, 3], [3]), ([0, 2, 3], [3]),
    ([0, 1, 2], [1, 2]), ([0, 1, 2], [True, 2]), ([0, 1, 2], [0, 3]),
    ([0, 1, 2], []), ([0., 1., 2.], [3]), ([False, True, True], [3]),
])
def test_malformed_boundaries_rejected_before_projection(positions, lengths):
    calls = []
    module = SimpleNamespace(qkv_proj=lambda x: calls.append(x))
    with pytest.raises(ValueError):
        segmented_attention_forward(module, torch.tensor(positions), torch.ones(3, 8), lengths,
                                    kernel=oracle)
    assert not calls


def test_cpu_default_and_wrong_kernel_shape_rejected():
    q, k, v, positions, lengths = fixture()
    with pytest.raises(ValueError, match='BF16 Ascend'):
        segmented_causal_attention(q, k, v, positions, lengths, scaling=.4)
    with pytest.raises(ValueError, match='incompatible output'):
        segmented_causal_attention(q, k, v, positions, lengths, scaling=.4,
                                   kernel=lambda *a, **kw: a[0][:1])
