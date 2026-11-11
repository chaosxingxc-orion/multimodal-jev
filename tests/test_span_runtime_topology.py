from types import SimpleNamespace

import pytest

from multimodal_jev.span_runtime_profile import PROFILE, verify_suite_admission
from multimodal_jev.span_runtime_topology import (
    TP4_DIAGNOSTIC, install_diagnostic_profile, runtime_topology,
)


@pytest.mark.parametrize('size,profile', [(4, PROFILE), (1, TP4_DIAGNOSTIC),
                                        (2, TP4_DIAGNOSTIC), (True, PROFILE)])
def test_incompatible_topology_is_rejected(size, profile):
    with pytest.raises(ValueError):
        runtime_topology(dict(tensor_parallel_size=size, runtime_profile=profile))


def test_tp4_diagnostic_cannot_enter_training_even_if_reported_passed():
    config = dict(tensor_parallel_size=4, runtime_profile=TP4_DIAGNOSTIC, passed=True)
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
