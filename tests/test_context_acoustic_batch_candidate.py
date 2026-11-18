from types import SimpleNamespace
import pytest
import torch
from tools.reports.context_acoustic_batch_candidate import audio_layout,batched_audio_forward
from multimodal_jev.span_audio_padding import short_audio_forward


class Attention(torch.nn.Module):
    def forward(self,x,bounds,maximum):
        # Real attention per segment makes cross-recording leakage observable.
        rows=[]
        for a,b in zip(bounds[:-1].tolist(),bounds[1:].tolist()):
            h=x[a:b]; rows.append(torch.softmax(h@h.T/2,-1)@h)
        return torch.cat(rows)


class Tower(torch.nn.Module):
    def __init__(self):
        super().__init__();self.n_window=50;self.n_window_infer=200
        self.conv2d1=torch.nn.Conv2d(1,2,3,2,1)
        self.conv2d2=torch.nn.Conv2d(2,2,3,2,1)
        self.conv2d3=torch.nn.Conv2d(2,2,3,2,1)
        self.conv_out=torch.nn.Linear(2,4)
        self.positional_embedding=SimpleNamespace(positional_embedding=torch.randn(13,4))
        self.layers=torch.nn.ModuleList([Attention()]);self.ln_post=torch.nn.LayerNorm(4)
        self.proj1=torch.nn.Linear(4,4);self.act=torch.nn.GELU();self.proj2=torch.nn.Linear(4,3)
    def _get_cnn_output_lengths(self,n):return (n+7)//8
    def compute_attn_mask_seqlen(self,b):return int((b[1:]-b[:-1]).max())


@pytest.mark.parametrize('lengths',[[1,7,77,99],[100,101,199,200,201,999],[77,100,305,9]])
def test_recording_boundary_order_and_gradient_equivalence(lengths):
    torch.manual_seed(12);tower=Tower().double()
    parts=[torch.randn(8,n,dtype=torch.double,requires_grad=True) for n in lengths]
    lens=torch.tensor(lengths);out=torch.tensor(audio_layout(lengths,100,200)[1])
    def compute(rows,ns,outs,batch):return batched_audio_forward(tower,torch.cat(rows,-1),ns,outs,conv_batch_chunks=batch)
    reference=torch.cat([compute([p],lens[i:i+1],out[i:i+1],1) for i,p in enumerate(parts)])
    actual=compute(parts,lens,out,4)
    torch.testing.assert_close(actual,reference,rtol=1e-10,atol=1e-10)
    ga=torch.autograd.grad(actual.square().sum(),parts+list(tower.parameters()),retain_graph=True)
    gr=torch.autograd.grad(reference.square().sum(),parts+list(tower.parameters()))
    for a,b in zip(ga,gr):torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-10)
    reversed_result=compute(parts[::-1],lens.flip(0),out.flip(0),3)
    torch.testing.assert_close(torch.cat(list(reversed(reversed_result.split(out.flip(0).tolist())))),actual)


def test_short_audio_matches_existing_full_window_reference():
    torch.manual_seed(1);tower=Tower().double();lens=torch.tensor([7,77,99]);out=(lens+7)//8
    features=torch.randn(8,int(lens.sum()),dtype=torch.double)
    expected=torch.cat([short_audio_forward(tower,p,lens[i:i+1],out[i:i+1])
        for i,p in enumerate(features.split(lens.tolist(),-1))])
    actual=batched_audio_forward(tower,features,lens,out,conv_batch_chunks=3)
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)


def test_chunk_and_attention_boundaries():
    chunks,out,attention=audio_layout([7,101,399],100,200)
    assert chunks==[7,100,1,100,100,100,99]
    assert out==[1,14,52] and attention==[1,14,26,26]
    with pytest.raises(ValueError):audio_layout([0],100,200)
    with pytest.raises(ValueError):audio_layout([1],100,150)


def test_invalid_output_lengths_rejected():
    tower=Tower()
    with pytest.raises(ValueError,match='boundaries'):
        batched_audio_forward(tower,torch.zeros(8,100),torch.tensor([100]),torch.tensor([12]),conv_batch_chunks=2)


def native_long_reference(tower, features):
    """Independent native per-recording pad/mask path for >=100 frames.

    Mirrors Qwen3OmniMoeAudioEncoder.forward's chunk pad_sequence, boolean
    extraction and attention windows; does not call candidate layout/forward.
    CPU layers are small numerical substitutes, not NPU kernels.
    """
    from torch.nn import functional as F
    frames = features.shape[-1]
    assert frames >= 100
    lengths = [100]*(frames//100) + ([frames % 100] if frames % 100 else [])
    chunks = features.T.split(lengths, dim=0)
    padded = torch.nn.utils.rnn.pad_sequence(chunks, batch_first=True).transpose(1, 2)
    valid = (torch.tensor(lengths)+7)//8
    mask = torch.arange(int(valid.max()))[None] < valid[:, None]
    hidden = padded[:, None]
    for conv in (tower.conv2d1, tower.conv2d2, tower.conv2d3):
        hidden = F.gelu(conv(hidden))
    b, c, f, t = hidden.shape
    hidden = tower.conv_out(hidden.permute(0, 3, 1, 2).contiguous().view(b, t, c*f))
    hidden = hidden + tower.positional_embedding.positional_embedding[:t].to(hidden.dtype)
    hidden = hidden[mask]
    width = mask.shape[-1]*(tower.n_window_infer//100)
    total = int(valid.sum()); segments = [width]*(total//width)
    if total % width: segments.append(total % width)
    bounds = torch.tensor([0, *segments], dtype=torch.int32).cumsum(0, dtype=torch.int32)
    for layer in tower.layers:
        hidden = layer(hidden, bounds, tower.compute_attn_mask_seqlen(bounds))
    return tower.proj2(tower.act(tower.proj1(tower.ln_post(hidden))))


@pytest.mark.parametrize('chunk_batch', [1, 4, 32])
def test_long_audio_matches_independent_native_pad_mask_and_gradients(chunk_batch):
    torch.manual_seed(17); tower = Tower().double()
    lengths = [100, 101, 199, 200, 201, 799]
    parts = [torch.randn(8, n, dtype=torch.double, requires_grad=True) for n in lengths]
    references = [native_long_reference(tower, part) for part in parts]
    expected = torch.cat(references)
    actual = batched_audio_forward(tower, torch.cat(parts, dim=-1), torch.tensor(lengths),
        torch.tensor([len(r) for r in references]), conv_batch_chunks=chunk_batch)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    params = parts + list(tower.parameters())
    ga = torch.autograd.grad(actual.square().sum(), params, retain_graph=True)
    gr = torch.autograd.grad(expected.square().sum(), params)
    for a, b in zip(ga, gr): torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-10)
