"""Audit-side compatibility with the frozen runtime's CPython 3.12 float sum.

Only ContextEventMetrics' module-global sum is scoped; no builtins, frozen
source, probabilities, comparison tolerances, or persisted results are changed.
Algorithm: CPython v3.12.13 Python/bltinmodule.c builtin_sum_impl, finite floats.
"""
import argparse
import builtins
from contextlib import contextmanager
import math
import runpy
import sys


def runtime_sum(values):
    values = list(values)
    if all(type(x) is int for x in values):
        return builtins.sum(values)
    if not values or any(type(x) is not float or not math.isfinite(x) for x in values):
        raise ValueError('Only integer counts or finite binary64 metric values supported')
    total, correction = values[0], 0.0
    for value in values[1:]:
        updated = total + value
        if abs(total) >= abs(value):
            correction += (total - updated) + value
        else:
            correction += (value - updated) + total
        total = updated
    if correction and math.isfinite(correction):
        total += correction
    if not math.isfinite(total):
        raise ValueError('Nonfinite metric sum')
    return total


@contextmanager
def runtime_metric_sum():
    from multimodal_jev import context_event_metrics as module
    if 'sum' in vars(module):
        raise ValueError('Unexpected metric sum override already installed')
    module.sum = runtime_sum
    try:
        yield
    finally:
        del module.sum


ALLOWED = {
    'tools.jev.audit_context_rl_selection_v2',
    'tools.jev.audit_context_rl_curve_v2',
    'tools.reports.compare_context_arithmetic_curves',
    'tools.reports.context_probability_endpoint_uncertainty',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', choices=sorted(ALLOWED), required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    sys.argv = [args.module, *forwarded]
    with runtime_metric_sum():
        runpy.run_module(args.module, run_name='__main__')


if __name__ == '__main__':
    main()
