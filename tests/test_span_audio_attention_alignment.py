import pytest
torch=pytest.importorskip('torch')
from multimodal_jev.span_audio_attention_alignment import windowed_audio_attention


def test_windowed_attention_matches_per_head_reference_and_isolates_windows():
    torch.manual_seed(17)
    q,k,v=[torch.randn(9,3,8) for _ in range(3)]
    bounds=torch.tensor([0,4,9],dtype=torch.int32)
    actual=windowed_audio_attention(q,k,v,bounds,8**-.5)
    expected=torch.empty(9,3,8)
    for start,end in ((0,4),(4,9)):
        for head in range(3):
            scores=(q[start:end,head]@k[start:end,head].T)*(8**-.5)
            expected[start:end,head]=scores.softmax(-1)@v[start:end,head]
    torch.testing.assert_close(actual,expected.reshape(9,-1))
    v[4:]+=100
    assert torch.equal(windowed_audio_attention(q,k,v,bounds,8**-.5)[:4],actual[:4])
    # Audio windows are noncausal: the first token can hear later audio.
    v[3]+=10
    assert not torch.equal(windowed_audio_attention(q,k,v,bounds,8**-.5)[0],actual[0])


@pytest.mark.parametrize('limits', [[1,9],[0,4,4,9],[0,8]])
def test_reject_incomplete_or_overlapping_window_description(limits):
    x=torch.zeros(9,2,4)
    with pytest.raises(ValueError,match='complete increasing'):
        windowed_audio_attention(x,x,x,torch.tensor(limits),.5)
