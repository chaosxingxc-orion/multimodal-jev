from types import SimpleNamespace
import pytest
torch=pytest.importorskip('torch')
from multimodal_jev.span_audio_padding import short_audio_forward


class Tower(torch.nn.Module):
    def __init__(self):
        super().__init__();self.n_window=50
        self.conv2d1=torch.nn.Conv2d(1,2,3,2,1)
        self.conv2d2=torch.nn.Conv2d(2,2,3,2,1)
        self.conv2d3=torch.nn.Conv2d(2,2,3,2,1)
        self.conv_out=torch.nn.Linear(2,3)
        self.positional_embedding=SimpleNamespace(positional_embedding=torch.randn(13,3))
        self.layers=[];self.ln_post=torch.nn.Identity();self.proj1=torch.nn.Identity()
        self.proj2=torch.nn.Identity();self.act=torch.nn.Identity()
    def _get_cnn_output_lengths(self,lens):
        for _ in range(3):lens=(lens+1)//2
        return lens
    def compute_attn_mask_seqlen(self,bounds):return int(bounds[-1])


@pytest.mark.parametrize('frames',[1,7,8,9,76,77,96,99])
def test_full_window_matches_reference_and_preserves_valid_output(frames):
    torch.manual_seed(9);tower=Tower();features=torch.randn(8,frames);length=torch.tensor([frames])
    valid=tower._get_cnn_output_lengths(length)
    actual=short_audio_forward(tower,features,length,valid)
    x=features.new_zeros(1,1,8,100);x[0,0,:,:frames]=features
    for conv in (tower.conv2d1,tower.conv2d2,tower.conv2d3):x=torch.nn.functional.gelu(conv(x))
    ref=tower.conv_out(x.permute(0,3,1,2).contiguous().reshape(1,13,2))
    ref=ref+tower.positional_embedding.positional_embedding
    assert torch.equal(actual,ref[0,:int(valid.item())])


def test_unpadded_convolutions_are_not_equivalent():
    torch.manual_seed(9);tower=Tower();features=torch.randn(8,77);length=torch.tensor([77]);valid=torch.tensor([10])
    actual=short_audio_forward(tower,features,length,valid)
    x=features[None,None]
    for conv in (tower.conv2d1,tower.conv2d2,tower.conv2d3):x=torch.nn.functional.gelu(conv(x))
    old=tower.conv_out(x.permute(0,3,1,2).contiguous().reshape(1,10,2))+tower.positional_embedding.positional_embedding[:10]
    assert (actual-old[0]).abs().max()>1e-5
