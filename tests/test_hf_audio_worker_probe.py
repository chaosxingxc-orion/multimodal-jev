import pytest
torch = pytest.importorskip('torch')
from multimodal_jev.hf_audio_worker_probe import unpack_audio_features


@pytest.mark.parametrize('sizes', [[43], [100], [243], [43, 243, 100]])
def test_packed_audio_order_padding_and_mask(sizes):
    source = torch.arange(2*sum(sizes)).reshape(2, -1).float()
    batch, mask = unpack_audio_features(source, torch.tensor(sizes), 100)
    assert batch.shape == (len(sizes), 2, ((max(sizes)+99)//100)*100)
    offset = 0
    for i, n in enumerate(sizes):
        assert torch.equal(batch[i, :, :n], source[:, offset:offset+n])
        assert (batch[i, :, n:] == 0).all()
        assert mask[i, :n].all() and not mask[i, n:].any()
        offset += n


@pytest.mark.parametrize('lengths', [[0], [-1], [5], [2.5, 1.5]])
def test_invalid_lengths_rejected(lengths):
    with pytest.raises(ValueError):
        unpack_audio_features(torch.zeros(2, 4), torch.tensor(lengths), 100)
