"""Candidate lifecycle and real identity transport, with substituted device I/O."""
from types import SimpleNamespace as NS
import pytest
pytest.importorskip('torch')
from multimodal_jev import span_batched_resident_rollout as m
from test_span_behavior_rollout import fixture as transport_fixture


def setup(monkeypatch, fault=None):
    original, inputs, calls, closed = transport_fixture(monkeypatch)
    original.config.update(runtime_profile=m.SHORT_AUDIO_PROFILE, candidate_profile=m.PROFILE,
        rollout_candidate_component='full_native_lora', physical_devices=[6])
    cfg = inputs()[0]
    events = []
    old_rpc = original.engine.llm_engine.collective_rpc
    def rpc(name):
        events.append(name)
        if name == 'align_jev_candidate_packed_attention':
            return [dict(attention='TND_sparse3_inner2', layers=28)]
        if name == 'align_jev_candidate_fixed_base':
            if fault == 'exception': raise RuntimeError('worker failed')
            result = dict(language_qkv=28, parallel_projections=['audio', 'language'],
                          tile_rows=128, tile_batch=8, lora='native_vllm_unchanged')
            if fault == 'tile': result['tile_rows'] = 64
            if fault == 'layers': result['language_qkv'] = 27
            return [result, result] if fault == 'tp' else [result]
        return old_rpc(name)
    original.engine.llm_engine.collective_rpc = rpc
    monkeypatch.setattr(m, 'validate_tp1', lambda config: events.append('device'))
    monkeypatch.setattr(m, 'checkpoint_identity', lambda path: (path, {}))
    monkeypatch.setattr(m, 'adapter_layout', lambda path: ['target'])
    monkeypatch.setattr(m, 'create_engine', lambda config, targets: original.engine)
    monkeypatch.setattr(m, 'install_rollout_profile', lambda engine, profile: events.append('base') or dict(profile=profile))
    return cfg, inputs, calls, closed, events


def test_candidate_install_and_switch_back_preserve_one_engine(monkeypatch):
    config, inputs, calls, closed, events = setup(monkeypatch)
    owner = m.BatchedResidentSpanRollout(config, 'A')
    assert events == ['device', 'base', 'align_jev_candidate_packed_attention', 'align_jev_candidate_fixed_base']
    reports = [owner.evaluate(*inputs(a)) for a in ('A', 'B', 'A')]
    assert [r['slot'] for r in reports] == [1, 2, 1]
    assert reports[0]['records'] == reports[2]['records']
    assert reports[0]['records'] != reports[1]['records']
    behavior = owner.rollout(*inputs('B')[:2])
    assert behavior['records'][0]['candidate_logprobs'] == reports[1]['records'][0]['candidate_logprobs']
    assert behavior['reference_compared'] is False
    owner.close(); owner.close(); assert len(closed) == 1


@pytest.mark.parametrize('field', ['runtime_profile', 'candidate_profile', 'rollout_candidate_component'])
def test_invalid_profile_rejected_before_device_access(monkeypatch, field):
    cfg, _, _, closed, events = setup(monkeypatch)
    cfg[field] = 'other'
    with pytest.raises(ValueError, match='complete candidate'): m.BatchedResidentSpanRollout(cfg, 'A')
    assert events == [] and closed == []


@pytest.mark.parametrize('fault', ['exception', 'tile', 'layers', 'tp'])
def test_failed_install_closes_engine(monkeypatch, fault):
    cfg, _, _, closed, _ = setup(monkeypatch, fault)
    with pytest.raises((ValueError, RuntimeError)): m.BatchedResidentSpanRollout(cfg, 'A')
    assert len(closed) == 1


@pytest.mark.parametrize('field', ['candidate_profile', 'rollout_candidate_component'])
def test_policy_switch_cannot_change_candidate_execution(monkeypatch, field):
    cfg, inputs, calls, _, _ = setup(monkeypatch)
    owner = m.BatchedResidentSpanRollout(cfg, 'A')
    policy, compiled, ref = inputs('B'); policy[field] = 'other'
    with pytest.raises(ValueError, match='runtime identity'): owner.evaluate(policy, compiled, ref)
    assert not calls and owner.phase == 'failed'
    with pytest.raises(ValueError, match='no retry'): owner.evaluate(*inputs())
