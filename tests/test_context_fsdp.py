from types import SimpleNamespace
import torch

from multimodal_jev.context_decision_head import ContextDecisionHead
from multimodal_jev.context_decision_model import ContextDecisionModel
from multimodal_jev.context_fsdp import context_fsdp_units


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=torch.nn.Embedding(100,8)
        self.proj=torch.nn.Linear(8,8)
    def forward(self,input_ids,**kwargs):
        return SimpleNamespace(last_hidden_state=self.proj(self.embedding(input_ids)))


def test_complete_head_unit_and_forward_hook_receive_gradients():
    model=ContextDecisionModel(Backbone(),ContextDecisionHead(8,2,16),99)
    units=context_fsdp_units(model)
    assert model.decision_head in units and model.backbone.proj in units
    assert not (units & (set(model.decision_head.modules())-{model.decision_head}))
    calls=[]
    handle=model.decision_head.register_forward_pre_hook(lambda *args:calls.append(True))
    ids=torch.tensor([[1,2,99]])
    z=model(dict(input_ids=ids,attention_mask=torch.ones_like(ids)),torch.tensor([[2]]),return_logits=True)
    z.sum().backward();handle.remove()
    assert calls==[True]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
