import pytest
import torch

from multimodal_jev.span_audio_padding import short_audio_forward
from test_span_audio_padding import Tower
from tools.reports.context_segmented_audio_candidate import segmented_audio_forward


def test_real_convolution_short_recordings_keep_boundaries_and_gradients():
    torch.manual_seed(317)
    tower = Tower()
    lengths = torch.tensor([1, 77, 99])
    after = tower._get_cnn_output_lengths(lengths)
    features = torch.randn(8, 177, requires_grad=True)
    calls = []
    def forward(x, f, a):
        calls.append((tuple(x.shape), f.tolist(), a.tolist(), x.is_contiguous()))
        return short_audio_forward(tower, x, f, a)
    actual = segmented_audio_forward(forward, features, lengths, after)
    expected = torch.cat([short_audio_forward(tower, x.contiguous(), lengths[i:i+1], after[i:i+1])
                          for i, x in enumerate(features.split(lengths.tolist(), dim=1))])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert [c[0] for c in calls] == [(8, 1), (8, 77), (8, 99)]
    assert all(c[3] for c in calls)
    gradient = torch.randn_like(actual)
    a = torch.autograd.grad(actual, features, gradient, retain_graph=True)[0]
    b = torch.autograd.grad(expected, features, gradient)[0]
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    changed = features.detach().clone(); changed[:, :78] += 100
    new = segmented_audio_forward(forward, changed, lengths, after)
    torch.testing.assert_close(new[-int(after[-1]):], actual[-int(after[-1]):], atol=0, rtol=0)
    order = [2, 0, 1]
    permuted = torch.cat([features.split(lengths.tolist(), dim=1)[i] for i in order], dim=1)
    new = segmented_audio_forward(forward, permuted, lengths[order], after[order])
    torch.testing.assert_close(new, torch.cat([actual.split(after.tolist())[i] for i in order]),
                               atol=0, rtol=0)


def test_mixed_lengths_delegate_existing_single_recording_behavior():
    # Does not implement or claim native long-audio convolution equivalence.
    lengths = torch.tensor([77, 100, 201], dtype=torch.int32)
    after = torch.tensor([10, 13, 27], dtype=torch.int32)
    x = torch.arange(8 * 378, dtype=torch.float32).reshape(8, 378)
    calls = []
    def forward(features, frames, tokens):
        calls.append((int(frames[0]), int(tokens[0])))
        return features[:, :int(tokens[0])].t()
    actual = segmented_audio_forward(forward, x, lengths, after)
    assert calls == [(77, 10), (100, 13), (201, 27)]
    assert actual.shape == (50, 8)
    torch.testing.assert_close(actual[10:23], x[:, 77:90].t(), atol=0, rtol=0)


@pytest.mark.parametrize('fault', ['nan', 'length', 'zero', 'float', 'bool', 'output_length', 'padded'])
def test_invalid_batch_rejected_before_tower(fault):
    features = torch.ones(8, 7); lengths = torch.tensor([3, 4]); after = torch.tensor([1, 1])
    if fault == 'nan': features[0, 0] = float('nan')
    if fault == 'length': lengths[0] = 2
    if fault == 'zero': lengths = torch.tensor([0, 7])
    if fault == 'float': lengths = lengths.float()
    if fault == 'bool': lengths = lengths.bool()
    if fault == 'output_length': after = torch.tensor([1])
    if fault == 'padded': features = features[None]
    calls = []
    with pytest.raises(ValueError):
        segmented_audio_forward(lambda *a: calls.append(a), features, lengths, after)
    assert not calls


def test_wrong_audio_output_boundaries_rejected():
    with pytest.raises(ValueError, match='token boundaries'):
        segmented_audio_forward(lambda *a: torch.ones(3, 8), torch.ones(8, 7),
                                torch.tensor([3, 4]), torch.tensor([1, 1]))
