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
