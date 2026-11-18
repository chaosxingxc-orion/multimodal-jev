"""Reseal the tail CLI fix without editing any frozen runtime dependencies."""
import argparse
import json
from types import FunctionType

from multimodal_jev.local_config import load_config
from tools.reports import context_macro_tail_operations as original
from tools.reports import prepare_context_macro_tail as preparation


def bind(function, **overrides):
    result = FunctionType(function.__code__, dict(function.__globals__, **overrides),
                          function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def command(c, role):
    argv = original.command(c, role)
    if role == 'client':
        old = c['remote_root']+'/package/tools/reports/context_macro_tail_candidate.py'
        if argv.count(old) != 1:
            raise ValueError('unique tail CLI anchor required')
        argv[argv.index(old)] = c['remote_root']+'/package/tools/reports/context_macro_tail_entry_v2.py'
    return argv


validate = bind(original.validate, command=command)
run = bind(original.run, validate=validate, command=command)
audit_collected = bind(original.audit_collected, validate=validate, command=command)
prepare = bind(preparation.prepare, validate=validate,
    ADDITIONS=(*preparation.ADDITIONS, 'tools/reports/context_macro_tail_entry_v2.py'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--local', required=True)
    p.add_argument('--action', choices=['prepare', 'validate', 'run', 'collect', 'audit'], required=True)
    args = p.parse_args(); c, provenance = load_config(args.config, args.local)
    if args.action == 'prepare': result = prepare(c, provenance)
    elif args.action == 'validate': result = validate(c)
    elif args.action == 'run': result = run(c, provenance)
    elif args.action == 'audit': result = audit_collected(c)
    else:
        from tools.reports.audit_context_macro_probe import collect_and_audit
        result = collect_and_audit(c, validator=validate, command_builder=command,
            output_auditor=original.audit_outputs, result_dirs=('run', 'journal', 'service-0', 'service-1'))
    print(json.dumps(result, indent=2))
