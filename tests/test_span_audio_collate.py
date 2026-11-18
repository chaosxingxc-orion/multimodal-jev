"""Contract/gradient tests; synthetic backbone, not Qwen/NPU admission."""
import copy
from types import SimpleNamespace

import pytest
import torch

from multimodal_jev.span_audio_collate import collate_span_audio, forward_span_audio_batch
from multimodal_jev.span_model import NativeAudioSpanModel


KW = dict(pad_token_id=0, audio_token_id=9, audio_chunk_size=100, audio_infer_window=800)


def item(frames, suffix=3):
    full, tail = divmod(frames, 100)
    count = full * 13 + (tail + 7) // 8
    ids = torch.tensor([[1] + [9] * count + [2] * suffix])
    width = ((frames + 99) // 100) * 100
    # Deliberately nonzero original feature padding: collation must preserve it.
    features = torch.arange(width * 2).reshape(1, 2, width).double() / 1000
    return dict(inputs=dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                            input_features=features,
                            input_features_mask=(torch.arange(width)[None] < frames).long()),
                positions=torch.tensor([[ids.shape[-1] - 1]]), candidates=[[2, 4, 7]])


class AudioBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(12, 4).double()
        self.calls = 0

    def forward(self, input_ids, attention_mask, input_features, input_features_mask,
                position_ids=None, **kwargs):
        self.calls += 1
        audio = (input_features * input_features_mask[:, None]).sum((1, 2))[:, None, None]
        pos = torch.arange(input_ids.shape[1])[None] if position_ids is None else position_ids
        hidden = self.emb(input_ids) + audio * .001 + pos[..., None] * .01
        return SimpleNamespace(last_hidden_state=(hidden * attention_mask[..., None]).cumsum(1))


def test_padding_preserves_features_order_and_span_locations_without_mutation():
    rows = [item(31), item(205, 5), item(100, 4)]
    before = copy.deepcopy(rows)
    inputs, positions, candidates = collate_span_audio(rows, **KW)
    assert inputs['input_ids'].shape[0] == 3
    assert inputs['input_features'].shape == (3, 2, 300)
    for i, row in enumerate(rows):
        for key, original in row['inputs'].items():
            assert torch.equal(inputs[key][i:i+1, ..., :original.shape[-1]], original)
            assert torch.equal(original, before[i]['inputs'][key])
        assert torch.equal(positions[i:i+1], row['positions'])
        length = row['inputs']['input_ids'].shape[-1]
        assert inputs['position_ids'][i, :length].tolist() == list(range(length))
        assert not inputs['attention_mask'][i, length:].any()
    assert candidates == rows[0]['candidates']


@pytest.mark.parametrize('size', [1, 2, 4, 8, 16, 32])
def test_one_forward_matches_serial_weighted_loss_gradient_and_adam_update(size):
    torch.manual_seed(317)
    model = NativeAudioSpanModel(SimpleNamespace(model=AudioBackbone(), lm_head=torch.nn.Linear(4, 12).double()))
    control = copy.deepcopy(model)
    rows = [item([31, 205, 100][i % 3], 3 + i % 2) for i in range(size)]
    weights = [0. if size > 1 and i == size - 1 else 1. + (i % 3) / 2 for i in range(size)]
    denom = sum(w != 0 for w in weights)
    actual = forward_span_audio_batch(model, rows, device='cpu', dtype=torch.float64, **KW)
    expected = [control(r['inputs'], r['positions'], r['candidates'])[1] for r in rows]
    assert model.backbone.calls == 1 and control.backbone.calls == size
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a[0], b[0], atol=2e-6, rtol=2e-6)
    def loss(values):
        return sum(-w * value[0][0, i % 3] / denom for i, (w, value) in enumerate(zip(weights, values)))
    la, lb = loss(actual), loss(expected)
    torch.testing.assert_close(la, lb, atol=2e-6, rtol=2e-6)
    la.backward(); lb.backward()
    for a, b in zip(model.parameters(), control.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-6)
    for net in (model, control):
        torch.optim.AdamW(net.parameters(), lr=1e-4).step()
    for a, b in zip(model.parameters(), control.parameters()):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize('fault', ['mask_hole', 'text_padding', 'audio_count', 'span_before_audio',
                                  'candidates', 'nan', 'position_ids', 'unknown_key'])
def test_bad_inputs_rejected(fault):
    rows = [item(31), item(205)]
    row = rows[0]
    if fault == 'mask_hole': row['inputs']['input_features_mask'][0, 1] = 0
    if fault == 'text_padding': row['inputs']['attention_mask'][0, -1] = 0
    if fault == 'audio_count': row['inputs']['input_ids'][0, 1] = 2
    if fault == 'span_before_audio': row['positions'][0, 0] = 0
    if fault == 'candidates': row['candidates'] = [[4, 2, 7]]
    if fault == 'nan': row['inputs']['input_features'][0, 0, 0] = float('nan')
    if fault == 'position_ids': row['inputs']['position_ids'] = torch.zeros_like(row['inputs']['input_ids'])
    if fault == 'unknown_key': row['inputs']['labels'] = torch.tensor([1])
    with pytest.raises(ValueError): collate_span_audio(rows, **KW)
