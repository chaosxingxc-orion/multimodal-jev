import torch

from tools.reports.evaluate_single_cards import bounded_batches


def test_long_audio_splits_before_hitting_padded_chunk_limit():
    def row(frames, valid, uid):
        return {'inputs': {'input_features': torch.empty(1, 1, frames),
                           'input_features_mask': (torch.arange(frames) < valid)[None, :]}, 'id': uid}

    groups = list(bounded_batches([[row(300, 290, 'short')]*3, [row(6800, 6700, 'long')]*24],
                                 width=128, chunk_size=100, max_chunks=768))
    assert max(len(batch)*max(item['inputs']['input_features'].shape[-1]//100 for item in batch)
               for batch in groups) <= 768
    assert sum(len(x) for x in groups) == 27
    assert all(sum(i['id'] == 'long' for i in batch) <= 11 for batch in groups)


def test_subchunk_audio_keeps_its_original_attention_geometry():
    def short(length):
        return {'inputs': {'input_features': torch.empty(1, 1, 100),
                           'input_features_mask': (torch.arange(100) < length)[None, :]}}

    items = [short(40), short(70), short(40)]
    groups = list(bounded_batches([items], 128, 100, 768))
    assert [[int(i['inputs']['input_features_mask'].sum()) for i in g] for g in groups] == [[40,40],[70]]


def test_matched_shape_splits_both_modalities_and_restores_original_order():
    from tools.reports.evaluate_single_cards import SingleCardScorer

    def item(tokens, frames, valid, ident):
        return {'id': ident, 'inputs': {
            'input_ids': torch.zeros(1, tokens),
            'input_features': torch.zeros(1, 1, frames),
            'input_features_mask': (torch.arange(frames) < valid)[None, :],
        }}

    group = [item(40, 200, 190, 0), item(60, 200, 190, 1),
             item(40, 200, 180, 2), item(40, 200, 190, 3)]
    calls = []
    scorer = object.__new__(SingleCardScorer)
    scorer.mode = 'matched_shape'

    def forward(_model, _device, _provider, _policy, rows):
        ids = [row['id'] for row in rows]
        calls.append(ids)
        return torch.tensor([[float(i)] for i in ids])

    scorer.forward = forward
    observed = scorer.evaluation_forward(None, None, None, None)(group)
    assert calls == [[0, 3], [1], [2]]
    assert observed.flatten().tolist() == [0., 1., 2., 3.]
