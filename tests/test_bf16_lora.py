import pytest
torch = pytest.importorskip('torch')
from multimodal_jev.bf16_lora import BF16MasterAdamW, cast_lora_bf16


def test_master_retains_updates_smaller_than_bf16_rounding():
    p = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    optimizer = BF16MasterAdamW([p], lr=1e-4)
    reference = torch.nn.Parameter(p.detach().float().clone())
    expected = torch.optim.AdamW([reference], lr=1e-4, weight_decay=0., foreach=False, fused=False)
    for _ in range(40):
        optimizer.zero_grad()
        p.grad = torch.ones_like(p)
        reference.grad = torch.ones_like(reference)
        expected.step()
        optimizer.step()
        torch.testing.assert_close(optimizer.masters[0], reference, rtol=0, atol=0)
        assert torch.equal(p, reference.to(torch.bfloat16))
    assert (p < 1).all()
    receipt = optimizer.receipt()
    assert receipt['execution_parameter_dtypes'] == ['torch.bfloat16']
    assert receipt['optimizer_tensor_dtypes'] == ['torch.float32']
    optimizer.zero_grad()
    assert p.grad is None and optimizer.masters[0].grad is None


def test_cast_rejects_unexpected_trainable_parameters_and_preserves_frozen_base():
    model = torch.nn.Module()
    model.base = torch.nn.Linear(2, 2).to(torch.bfloat16).requires_grad_(False)
    model.lora_A = torch.nn.Linear(2, 1, bias=False)
    original = model.base.weight.detach().clone()
    assert cast_lora_bf16(model) == ['lora_A.weight']
    assert model.lora_A.weight.dtype == torch.bfloat16
    assert torch.equal(model.base.weight, original)
    model.base.requires_grad_(True)
    with pytest.raises(ValueError, match='only LoRA'):
        cast_lora_bf16(model)


def test_nonfinite_gradient_cannot_update_execution_weights():
    p = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
    optimizer = BF16MasterAdamW([p], lr=1e-4)
    p.grad = torch.full_like(p, float('nan'))
    with pytest.raises(ValueError, match='nonfinite'):
        optimizer.step()
    assert p.item() == 1


def test_peft_reload_explicitly_disables_adapter_upcast(monkeypatch):
    import sys
    from types import SimpleNamespace
    from multimodal_jev.sft import initialize_adapter
    model = torch.nn.Module()
    model.lora_A = torch.nn.Linear(2, 1, bias=False)
    calls = []
    def load(base, path, **kwargs):
        calls.append(kwargs)
        return model
    monkeypatch.setitem(sys.modules, 'peft', SimpleNamespace(
        PeftModel=SimpleNamespace(from_pretrained=load), LoraConfig=None, get_peft_model=None))
    actual = initialize_adapter(object(), {'initial_adapter':'fixture',
                                          'lora_execution_dtype':'bfloat16'})
    assert actual is model
    assert calls == [{'is_trainable':True, 'autocast_adapter_dtype':False}]
    assert model.lora_A.weight.dtype == torch.bfloat16


def test_resume_preserves_sub_bf16_updates_and_adam_moments():
    p=torch.nn.Parameter(torch.ones(2,dtype=torch.bfloat16))
    optimizer=BF16MasterAdamW([p],lr=1e-4)
    for step in range(17):
        optimizer.zero_grad();p.grad=torch.tensor([1.,-.5],dtype=p.dtype);optimizer.step()
    saved=optimizer.state_dict()
    q=torch.nn.Parameter(p.detach().clone())
    restored=BF16MasterAdamW([q],lr=.1)
    restored.load_state_dict(saved)
    assert restored.optimizer.param_groups[0]['lr']==1e-4
    for step in range(30):
        grad=torch.tensor([1.+step/30.,-.5],dtype=p.dtype)
        optimizer.zero_grad();p.grad=grad.clone();optimizer.step()
        restored.zero_grad();q.grad=grad.clone();restored.step()
        assert torch.equal(p,q)
        assert torch.equal(optimizer.masters[0],restored.masters[0])
    bad=BF16MasterAdamW([torch.nn.Parameter(torch.zeros_like(p))],lr=1e-4)
    with pytest.raises(ValueError,match='differs'):
        bad.load_state_dict(saved)
