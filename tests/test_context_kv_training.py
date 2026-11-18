import copy
import pytest
from test_context_macro_training_operations import fixture
from tools.reports import context_kv_training_operations as kv
from tools.reports import context_acoustic_training_operations as acoustic


def test_capacity_change_retains_runtime_identity_and_policy():
    original = {'policy': {'kv_cache_memory_bytes': 2**30, 'policy_id': 'fixed'},
                'pins': {'/model/config': 'hash'}, 'dual_runtime_identity': {'source': 'sealed'}}
    candidate = copy.deepcopy(original)
    candidate['policy']['kv_cache_memory_bytes'] = 16 * 2**30
    kv.check_delta(original, candidate, service=True, kv_bytes=16*2**30)
    for mutate in (
        lambda c: c['policy'].update(policy_id='different'),
        lambda c: c['pins'].update({'/model/config': 'different'}),
        lambda c: c['dual_runtime_identity'].update(source='different'),
        lambda c: c['policy'].update(max_batch_queries=48),
    ):
        broken = copy.deepcopy(candidate); mutate(broken)
        with pytest.raises(ValueError, match='only service KV'):
            kv.check_delta(original, broken, service=True, kv_bytes=16*2**30)
    assert original['policy']['kv_cache_memory_bytes'] == 2**30


def test_actor_settings_must_be_identical():
    original = {'global_batch': 128, 'microbatch': 8, 'algorithm': 'locked', 'pins': {'a': 'b'}}
    kv.check_delta(original, copy.deepcopy(original), service=False, kv_bytes=16*2**30)
    with pytest.raises(ValueError):
        kv.check_delta(original, dict(original, microbatch=16), service=False, kv_bytes=16*2**30)


@pytest.mark.parametrize('role', ['client','preflight','service-0','service-1'])
def test_commands_reuse_acoustic_lifecycle(tmp_path, role):
    c, _ = fixture(tmp_path)
    assert kv.command(c, role) == acoustic.command(c, role)


def test_prepared_candidate_does_not_dispatch(monkeypatch):
    monkeypatch.setattr(kv.shared, 'run_services', lambda *a, **k: pytest.fail('no dispatch'))
    with pytest.raises(ValueError): kv.run({'finite_training_dispatch_allowed': False}, {})
