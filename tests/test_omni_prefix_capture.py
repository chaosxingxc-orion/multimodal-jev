import pytest
torch=pytest.importorskip('torch')
from multimodal_jev.omni_prefix_capture import PrefixCaptured,capture_prefix


class Layer(torch.nn.Module):
    def __init__(self):super().__init__();self.calls=0
    def forward(self,hidden_states,**kwargs):self.calls+=1;return hidden_states+1


def test_capture_stops_before_layer_and_removes_hook():
    layer=Layer().eval();mask=torch.tensor([[0,1,1],[1,1,1]])
    hidden=torch.arange(24.).reshape(2,3,4);out={}
    with torch.inference_mode():
        with pytest.raises(PrefixCaptured),capture_prefix(layer,mask,out):
            layer(hidden,position_embeddings=(hidden,hidden+2),attention_mask=None)
        assert layer.calls==0 and torch.equal(out['hidden_states'],hidden)
        assert out['hidden_states'].data_ptr()!=hidden.data_ptr()
        assert torch.equal(out['position_embeddings'][1],hidden+2)
        layer(hidden)
        assert layer.calls==1 and not layer._forward_pre_hooks


def test_bad_shape_does_not_become_successful_capture():
    layer=Layer().eval();out={}
    with torch.inference_mode():
        with pytest.raises(ValueError,match='hidden states'),capture_prefix(layer,torch.ones(2,3),out):
            layer(torch.zeros(1,3,4))
    assert not out and layer.calls==0 and not layer._forward_pre_hooks


def test_capture_refuses_grad_or_reuse():
    layer=Layer().eval()
    with pytest.raises(ValueError,match='eval inference'),capture_prefix(layer,torch.ones(1,1),{}):pass
    with torch.inference_mode():
        with pytest.raises(ValueError,match='empty capture'),capture_prefix(layer,torch.ones(1,1),{'old':1}):pass
