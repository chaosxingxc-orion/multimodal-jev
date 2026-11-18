from types import SimpleNamespace

import pytest
import torch

from multimodal_jev import span_batch_rollout_candidate as candidate


def test_rollout_layout_and_native_projection_flow(monkeypatch):
    q = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    k = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    v = k + 50
    hidden = torch.zeros(3, 8)
    positions = torch.arange(3)
    seen = []

    def attention(query, key, value, lengths, *, scaling, kernel):
        assert query.shape == (1, 4, 3, 2)
        assert key.shape == value.shape == (1, 2, 3, 2)
        assert lengths == [3] and scaling == .5
        assert torch.equal(query[0].transpose(0, 1).reshape(3, 8), q + 11)
        assert torch.equal(key[0].transpose(0, 1).reshape(3, 4), k + 22)
        assert torch.equal(value[0].transpose(0, 1).reshape(3, 4), v)
        assert kernel is candidate.masked_row_varlen_kernel
        return query.transpose(1, 2)

    def project(x):
        seen.append(x.clone())
        return x * 3, None

    module = SimpleNamespace(
        qkv_proj=lambda x: (torch.cat((q, k, v), -1), None),
        q_size=8, kv_size=4, num_heads=4, num_kv_heads=2, head_dim=2,
        q_norm=lambda x: x + 1, k_norm=lambda x: x + 2,
        rotary_emb=lambda p, a, b: (a + 10, b + 20), scaling=.5, o_proj=project)
    monkeypatch.setattr(candidate, 'packed_causal_attention', attention)
    result = candidate.packed_attention_forward(module, positions, hidden)
    assert len(seen) == 1 and torch.equal(result, (q + 11) * 3)
    with pytest.raises(ValueError, match='complete uncached'):
        candidate.packed_attention_forward(module, positions + 1, hidden)


def test_install_rejects_partial_tp_mutation():
    first = SimpleNamespace(qkv_proj=SimpleNamespace(tp_size=1))
    second = SimpleNamespace(qkv_proj=SimpleNamespace(tp_size=4))
    model = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(
        layers=[SimpleNamespace(self_attn=first), SimpleNamespace(self_attn=second)])))
    with pytest.raises(ValueError, match='TP1'):
        candidate.install_packed_rollout_attention(model)
    assert not hasattr(first, '_jev_packed_candidate')
    second.qkv_proj.tp_size = 1
    receipt = candidate.install_packed_rollout_attention(model)
    assert receipt['layers'] == 2 and not receipt['production_promoted']
    with pytest.raises(ValueError, match='fresh'):
        candidate.install_packed_rollout_attention(model)
