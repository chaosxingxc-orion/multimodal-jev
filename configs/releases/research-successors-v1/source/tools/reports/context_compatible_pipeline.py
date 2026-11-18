"""Scoped host-only metric compatibility for unchanged experiment drivers.

The published NPU packages and original invocation functions remain unchanged
on disk. Only selected host audit subprocesses use the metric compatibility
entry point. No global subprocess or builtins monkey patch is installed.
"""
import argparse
from contextlib import contextmanager
import importlib
from pathlib import Path
import runpy
import subprocess
import sys

from tools.reports.context_metric_python312 import runtime_metric_sum


DIRECT = frozenset({
    'tools.jev.audit_context_rl_selection_v2',
    'tools.jev.audit_context_rl_curve_v2',
    'tools.reports.compare_context_arithmetic_curves',
    'tools.reports.prepare_fsc_arithmetic_source',
    'tools.reports.context_probability_endpoint_uncertainty',
})
DRIVERS = {
    'tools.reports.run_context_arithmetic_experiment': {
        'audit_context_rl_selection_v2': 'tools.jev.audit_context_rl_selection_v2',
        'audit_context_rl_curve_v2': 'tools.jev.audit_context_rl_curve_v2',
    },
    'tools.reports.run_fsc_arithmetic_experiment': {
        'prepare_fsc_context_source_v2': 'tools.reports.prepare_fsc_arithmetic_source',
    },
}


@contextmanager
def compatible_invocations(module_name):
    module = importlib.import_module(module_name)
    original = module.invoke
    routes = DRIVERS[module_name]
    def invoke(name, arguments, log):
        if name not in routes:
            return original(name, arguments, log)
        with Path(log).open('x') as stream:
            subprocess.run([sys.executable, '-m', 'tools.reports.context_compatible_pipeline', '--module', routes[name], '--',
                *map(str, arguments)], stdout=stream, stderr=subprocess.STDOUT, check=True)
    module.invoke = invoke
    try:
        yield module
    finally:
        module.invoke = original


def execute(module_name, arguments):
    if module_name in DIRECT:
        previous = sys.argv
        try:
            sys.argv = [module_name, *arguments]
            with runtime_metric_sum():
                runpy.run_module(module_name, run_name='__main__')
        finally:
            sys.argv = previous
        return
    if module_name not in DRIVERS:
        raise ValueError('Unregistered host compatibility route')
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args(arguments)
    with compatible_invocations(module_name) as driver:
        return driver.run(driver.read(args.spec), Path(args.output), launch=args.launch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', choices=sorted(DIRECT | DRIVERS.keys()), required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    execute(args.module, forwarded)
