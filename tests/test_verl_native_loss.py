"""Admission checks; numerical loop equivalence is tested in the pinned image."""
import sys
from types import SimpleNamespace as NS
import pytest

torch = pytest.importorskip('torch')
from multimodal_jev.verl_native_loss import native_token_mean_loss, register_native_token_mean_loss, LOSS_NAME


def setup(monkeypatch):
    calls = []
    def forbidden(**kwargs):
        calls.append(kwargs)
        raise AssertionError('invalid config reached upstream loss')
    monkeypatch.setitem(sys.modules, 'verl.trainer.ppo.core_algos',
                        NS(compute_policy_loss_vanilla=forbidden))
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 1)
    config = NS(use_dynamic_bsz=False, ppo_epochs=1, use_kl_loss=False,
                entropy_coeff=0., calculate_entropy=False, use_rollout_log_probs=True,
                global_batch_info={}, ppo_mini_batch_size=8, ppo_micro_batch_size_per_gpu=2,
                jev_batch_num_tokens=24, jev_batch_num_responses=8)
    return config, calls


@pytest.mark.parametrize('field,value', [
    ('use_dynamic_bsz', True), ('ppo_epochs', 2), ('use_kl_loss', True),
    ('entropy_coeff', .1), ('calculate_entropy', True), ('use_rollout_log_probs', False),
    ('global_batch_info', {'batch_num_tokens': 24}), ('ppo_micro_batch_size_per_gpu', 3),
    ('jev_batch_num_responses', 16), ('jev_batch_num_tokens', 0),
])
def test_unsupported_contract_rejected_before_loss(monkeypatch, field, value):
    config, calls = setup(monkeypatch)
    setattr(config, field, value)
    mask = torch.ones(2, 3)
    with pytest.raises(ValueError):
        native_token_mean_loss(mask, mask, mask, mask, config=config)
    assert not calls


@pytest.mark.parametrize('kind', ['rank', 'fractional_mask', 'empty_row', 'wrong_rows', 'is_weights'])
def test_invalid_execution_rejected(monkeypatch, kind):
    config, calls = setup(monkeypatch)
    mask = torch.ones(2, 3)
    kwargs = {}
    if kind == 'rank': monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
    elif kind == 'fractional_mask': mask[0, 0] = .5
    elif kind == 'empty_row': mask[0] = 0
    elif kind == 'wrong_rows': mask = torch.ones(1, 3)
    else: kwargs['rollout_is_weights'] = torch.ones_like(mask)
    with pytest.raises(ValueError):
        native_token_mean_loss(mask, mask, mask, mask, config=config, **kwargs)
    assert not calls


def test_registration_is_idempotent_and_does_not_overwrite(monkeypatch):
    registry = {}
    monkeypatch.setitem(sys.modules, 'verl.trainer.ppo.core_algos', NS(
        POLICY_LOSS_REGISTRY=registry,
        register_policy_loss=lambda name: lambda function: registry.setdefault(name, function)))
    assert register_native_token_mean_loss() == LOSS_NAME
    assert register_native_token_mean_loss() == LOSS_NAME
    registry[LOSS_NAME] = object()
    with pytest.raises(ValueError, match='collision'): register_native_token_mean_loss()
