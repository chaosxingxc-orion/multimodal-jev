import json
import pytest
torch = pytest.importorskip('torch')
from multimodal_jev.bf16_lora import BF16MasterAdamW
from multimodal_jev.span_checkpoint import save_span_checkpoint, restore_span_optimizer


class FixtureAdapter(torch.nn.Module):
    """Exercise checkpoint transactions; not a safetensors or PEFT test."""
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
        self.b = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))

    def save_pretrained(self, destination):
        destination.mkdir()
        # Storage is opaque to the manifest; runtime PEFT owns actual encoding.
        (destination/'adapter_model.safetensors').write_bytes(b'fixture-opaque-weights')
        (destination/'adapter_model.safetensors').chmod(0o600)
        (destination/'adapter_config.json').write_text('{}')


def test_checkpoint_roundtrip_authentication_and_version_exclusion(tmp_path):
    model = FixtureAdapter()
    optimizer = BF16MasterAdamW(model.parameters(), lr=.001)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    optimizer.step()
    target = tmp_path/'policy-1'
    policy = save_span_checkpoint(target, model, model, optimizer, 'base-hash', 1)
    assert policy.startswith('base-hash:') and not (tmp_path/'policy-1.pending').exists()
    assert (target/'adapter/adapter_model.safetensors').stat().st_mode & 0o777 == 0o644
    assert (target/'adapter').stat().st_mode & 0o777 == 0o777
    restored = BF16MasterAdamW(model.parameters(), lr=.1)
    receipt = restore_span_optimizer(target, model, restored, 'base-hash')
    assert receipt['completed_updates'] == 1
    assert restored.receipt()['optimizer_state_parameters'] == 2
    assert restored.receipt()['optimizer_steps'] == [1]
    for saved, loaded in zip(optimizer.optimizer.state.values(), restored.optimizer.state.values(), strict=True):
        assert saved.keys() == loaded.keys()
        assert all(torch.equal(saved[k], loaded[k]) for k in saved)
    assert all(torch.equal(a,b) for a,b in zip(optimizer.masters, restored.masters))
    with pytest.raises(ValueError, match='already exists'):
        save_span_checkpoint(target, model, model, optimizer, 'base-hash', 1)
    with pytest.raises(ValueError, match='identity mismatch'):
        restore_span_optimizer(target, model, restored, 'another-base')
    reversed_optimizer = BF16MasterAdamW(list(model.parameters())[::-1], lr=.1)
    with pytest.raises(ValueError, match='order'):
        restore_span_optimizer(target, model, reversed_optimizer, 'base-hash')
    (target/'adapter/adapter_model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='identity mismatch'):
        restore_span_optimizer(target, model, restored, 'base-hash')


def test_reject_optimizer_tampering_before_mutation(tmp_path):
    model = FixtureAdapter()
    optimizer = BF16MasterAdamW(model.parameters(), lr=.001)
    target = tmp_path/'policy-1'
    save_span_checkpoint(target, model, model, optimizer, 'base', 0)
    (target/'optimizer.pt').write_bytes(b'changed')
    before = [p.clone() for p in optimizer.masters]
    with pytest.raises(ValueError, match='identity mismatch'):
        restore_span_optimizer(target, model, optimizer, 'base')
    assert all(torch.equal(a,b) for a,b in zip(before, optimizer.masters))


def test_cursor_is_bound_to_optimizer_checkpoint(tmp_path):
    model=FixtureAdapter();optimizer=BF16MasterAdamW(model.parameters(),lr=.001)
    target=tmp_path/'policy'
    state=dict(batch_index=0,next_batch_index=1,schedule_sha256='schedule')
    save_span_checkpoint(target,model,model,optimizer,'base',1,training_state=state)
    assert restore_span_optimizer(target,model,optimizer,'base')['training_state']==state
    receipt=json.loads((target/'CHECKPOINT.json').read_text())
    receipt['training_state']['next_batch_index']=4
    (target/'CHECKPOINT.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='lineage mismatch'):
        restore_span_optimizer(target,model,optimizer,'base')
