import pytest
torch=pytest.importorskip('torch')
from multimodal_jev.omni_activation_trace import PaddingTrace


class Attention(torch.nn.Module):
    def forward(self,x):return x,None


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__();self.self_attn=Attention();self.mlp=torch.nn.Identity()
    def forward(self,x):return self.mlp(self.self_attn(x)[0])


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layers=torch.nn.ModuleList([Layer()])
    def forward(self,x):return self.layers[0](x)


def test_mask_aligns_valid_positions_and_reference_is_immutable(tmp_path):
    model=Model().eval();trace=PaddingTrace(tmp_path/'trace')
    x=torch.arange(12.).reshape(1,3,4);mask=torch.ones(1,3)
    with torch.inference_mode():
        with trace.capture(model,mask):model(x)
        padded=torch.cat([torch.ones(1,2,4)*99,x],1)
        with trace.capture(model,torch.tensor([[0,0,1,1,1]]),compare=True):model(padded)
        assert len(trace.reports)==4 and all(r['equal'] for r in trace.reports)
        first=next(iter(trace.reference));(trace.directory/(first+'.pt')).write_bytes(b'corrupt')
        with pytest.raises(ValueError,match='changed'):
            with trace.capture(model,mask,compare=True):model(x)
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


@pytest.mark.parametrize('failure',['budget','forward','incomplete'])
def test_failure_removes_hooks_and_cannot_be_accepted(tmp_path,failure):
    model=Model().eval();trace=PaddingTrace(tmp_path/'trace',max_bytes=1 if failure=='budget' else 1000)
    with torch.inference_mode(),pytest.raises((ValueError,RuntimeError)):
        with trace.capture(model,torch.ones(1,3)):
            if failure=='forward':raise RuntimeError('forward failed')
            if failure=='budget':model(torch.ones(1,3,4))
    assert not trace.complete and not trace.active
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


def test_training_or_autograd_trace_rejected(tmp_path):
    model=Model();trace=PaddingTrace(tmp_path/'trace')
    with torch.inference_mode(),pytest.raises(ValueError):
        with trace.capture(model,torch.ones(1,3)):pass
    model.eval()
    with pytest.raises(ValueError):
        with trace.capture(model,torch.ones(1,3)):pass
