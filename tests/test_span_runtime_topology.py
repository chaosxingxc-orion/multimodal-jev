from types import SimpleNamespace

import pytest

from multimodal_jev.span_runtime_profile import PROFILE, verify_suite_admission
from multimodal_jev.span_runtime_topology import (
    TP4_DIAGNOSTIC, TP4_HF_ORDER, TP4_COMPLETE_K, TP4_FULL_WIDTH, install_diagnostic_profile, runtime_topology,
)


@pytest.mark.parametrize('size,profile', [(4, PROFILE), (1, TP4_DIAGNOSTIC),
                                        (2, TP4_DIAGNOSTIC), (True, PROFILE)])
def test_incompatible_topology_is_rejected(size, profile):
    with pytest.raises(ValueError):
        runtime_topology(dict(tensor_parallel_size=size, runtime_profile=profile))


@pytest.mark.parametrize('devices', [[0],[3],[8],[True],[4,5],[],[4,4],(5,)])
def test_single_card_requires_one_authorized_physical_device(devices):
    with pytest.raises(ValueError):runtime_topology(dict(runtime_profile=PROFILE,physical_devices=devices))


def test_single_card_can_use_authorized_idle_card_without_changing_architecture():
    original=runtime_topology(dict(runtime_profile=PROFILE))
    idle=runtime_topology(dict(runtime_profile=PROFILE,physical_devices=[5]))
    assert original['devices']==['davinci4'] and idle['devices']==['davinci5']
    assert {k:v for k,v in original.items() if k!='devices'}=={k:v for k,v in idle.items() if k!='devices'}


@pytest.mark.parametrize('profile', [TP4_DIAGNOSTIC, TP4_HF_ORDER, TP4_COMPLETE_K, TP4_FULL_WIDTH])
def test_tp4_diagnostic_cannot_enter_training_even_if_reported_passed(profile):
    config = dict(tensor_parallel_size=4, runtime_profile=profile, passed=True)
    assert runtime_topology(config)['diagnostic_only']
    with pytest.raises(ValueError, match='unsupported Span runtime profile'):
        verify_suite_admission(config)
    assert runtime_topology(dict(runtime_profile=PROFILE))['size'] == 1


@pytest.mark.parametrize('receipts', [[{'after': 'disable'}] * 3,
                                     [{'after': 'disable'}] * 3 + [{'after': 'enable'}]])
def test_all_workers_must_confirm_configuration(receipts):
    engine = SimpleNamespace(llm_engine=SimpleNamespace(collective_rpc=lambda rpc: receipts))
    with pytest.raises(ValueError, match='four internal-format receipts required'):
        install_diagnostic_profile(engine, TP4_DIAGNOSTIC)


def test_tp4_alignment_requires_all_workers_and_runs_after_format():
    calls = []
    def rpc(name):
        calls.append(name)
        if name == 'align_jev_internal_format':
            return [{'after': 'disable'}] * 4
        return [{'tensor_parallel_size': 4, 'diagnostic_only': True}] * 3
    engine = SimpleNamespace(llm_engine=SimpleNamespace(collective_rpc=rpc))
    with pytest.raises(ValueError, match='four TP4 alignment receipts'):
        install_diagnostic_profile(engine, TP4_HF_ORDER)
    assert calls == ['align_jev_internal_format', 'align_jev_tp4_hf_order']
