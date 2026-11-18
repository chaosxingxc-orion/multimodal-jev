import builtins
import math

import pytest

from tools.reports.context_metric_python312 import runtime_sum, runtime_metric_sum
from tools.jev.context_metric_equivalence import compare


def test_compensated_float_and_exact_integer_counts():
    assert runtime_sum([1e16,1.,-1e16]) == 1.
    assert runtime_sum([2**70,3,7]) == 2**70+10
    assert runtime_sum([]) == 0
    assert runtime_sum([.1]*10) == 1.


@pytest.mark.parametrize('values',[[1.,True],[1.,1],[float('nan')],[float('inf')]])
def test_reject_unsupported(values):
    with pytest.raises(ValueError):runtime_sum(values)


def test_scope_and_rejection_threshold_unchanged():
    from multimodal_jev import context_event_metrics as module
    original=builtins.sum
    with pytest.raises(RuntimeError):
        with runtime_metric_sum():
            assert module.sum is runtime_sum and builtins.sum is original
            with pytest.raises(ValueError,match='override'):
                with runtime_metric_sum():pass
            raise RuntimeError('exit')
    assert 'sum' not in vars(module) and builtins.sum is original
    with pytest.raises(ValueError,match='metric differs'):
        compare({'metrics':{'top1_macro_f1':.75}}, {'metrics':{'top1_macro_f1':math.nextafter(.75,1.)}})
