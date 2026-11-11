import pytest
torch=pytest.importorskip('torch')
from multimodal_jev.span_attention_alignment import eager_causal_attention,aligned_attention_forward


def test_grouped_attention_matches_independent_head_reference_and_blocks_future():
    torch.manual_seed(17)
    q=torch.randn(7,4,8);k=torch.randn(7,2,8);v=torch.randn(7,2,8)
    actual=eager_causal_attention(q,k,v,8**-.5)
    expected=torch.empty_like(actual)
    for t in range(7):
        for h in range(4):
            scores=(q[t,h]@k[:t+1,h//2].T)*(8**-.5)
            expected[t,h]=scores.softmax(-1)@v[:t+1,h//2]
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
    k[4:]+=10;v[4:]*=5
    torch.testing.assert_close(eager_causal_attention(q,k,v,8**-.5)[:4],actual[:4])


def test_decode_and_packed_requests_rejected_before_projection():
    for positions in (torch.tensor([5]),torch.tensor([0,1,0,1])):
        with pytest.raises(ValueError,match='single complete'):
            aligned_attention_forward(object(),positions,torch.ones(len(positions),8))
