"""Actual head+LoRA update/recovery; synthetic backbone, no task-gain claim."""
import copy

import torch

from multimodal_jev.bf16_lora import BF16MasterAdamW
from multimodal_jev.context_decision_model import ContextDecisionModel
from multimodal_jev.context_decision_head import ContextDecisionHead
from test_context_policy_snapshot import Backbone


def test_bf16_context_update_fp32_resume_matches_uninterrupted_second_update():
    torch.manual_seed(317)
    model = ContextDecisionModel(Backbone(),ContextDecisionHead(8,2,16),19).bfloat16().eval()
    # Match actual PEFT initialization; B changes first, A receives gradient later.
    with torch.no_grad(): model.backbone.q_proj.lora_B['default'].weight.zero_()
    names = [n for n,p in model.named_parameters() if p.requires_grad]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = BF16MasterAdamW(params,lr=1e-3)
    inputs = dict(input_ids=torch.tensor([[1,2,19,3,19]]),attention_mask=torch.ones(1,5,dtype=torch.long))
    pos = torch.tensor([[2,4]])
    initial = {n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}

    def update(m,o):
        o.zero_grad()
        logits = m.score_logits(inputs,pos)
        (logits*torch.tensor([[-.5,.5]])).sum().backward()
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.)
        o.step()

    update(model,opt)
    assert not torch.equal(model.backbone.q_proj.lora_B['default'].weight,
                           initial['backbone.q_proj.lora_B.default.weight'])
    assert any(not torch.equal(p,initial[n]) for n,p in model.named_parameters() if n.startswith('decision_head.'))
    restored = copy.deepcopy(model)
    resume = BF16MasterAdamW([p for p in restored.parameters() if p.requires_grad],lr=1e-3)
    assert [n for n,p in restored.named_parameters() if p.requires_grad] == names
    resume.load_state_dict(opt.state_dict())
    assert torch.equal(model(inputs,pos),restored(inputs,pos))
    update(model,opt); update(restored,resume)
    for (name,p),(other,q) in zip(model.named_parameters(),restored.named_parameters(),strict=True):
        assert name == other and torch.equal(p,q)
    assert model.backbone.q_proj.lora_A['default'].weight.grad.abs().sum() > 0
    for a,b in zip(opt.masters,resume.masters,strict=True):
        assert a.dtype == b.dtype == torch.float32 and torch.equal(a,b)
    for key,values in opt.optimizer.state_dict()['state'].items():
        for field,value in values.items():
            got = resume.optimizer.state_dict()['state'][key][field]
            if torch.is_tensor(value):
                assert value.dtype == got.dtype == torch.float32 and torch.equal(value,got)
