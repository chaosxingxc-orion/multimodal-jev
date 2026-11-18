"""Formal634 collection adapter retaining the measured acoustic service entry.

No preparation, selection or dispatch: full matched100 and matching tail/parity
admission remain required before this runtime can become a training baseline.
"""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from tools.reports import context_macro_formal_operations as formal
from tools.reports import context_acoustic_training_operations as acoustic
from tools.reports.watch_context_acoustic_training import bind


def command(c, role):
    argv = formal.command(c, role)
    if role.startswith('service-'):
        old = 'tools.reports.context_dual_runtime_candidate'
        if argv.count(old) != 1:
            raise ValueError('unique formal acoustic service anchor required')
        argv[argv.index(old)] = 'tools.reports.context_acoustic_service_candidate'
    return argv


_validate = bind(formal.validate, command=command)
audit_outputs = formal.audit_outputs


def validate(c):
    receipt = _validate(c)
    package = Path(c['prepared'])/'package'
    for name in acoustic.ADDITIONS:
        path = Path('tools/reports')/name
        if file_hash(package/path) != file_hash(Path(c['project'])/path):
            raise ValueError('declared acoustic implementation differs: '+name)
    return receipt


def audit_collected(c):
    from tools.reports.audit_context_macro_training import audit_collected as audit
    return audit(c, validator=validate, command_builder=command, output_auditor=audit_outputs)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--local', required=True)
    p.add_argument('--action', choices=['validate','collect','audit'], default='validate')
    args = p.parse_args(); c, _ = load_config(args.config,args.local)
    if args.action == 'validate': result = validate(c)
    elif args.action == 'audit': result = audit_collected(c)
    else:
        from tools.reports.audit_context_macro_probe import collect_and_audit
        result = collect_and_audit(c, validator=validate, command_builder=command,
            output_auditor=audit_outputs, result_dirs=('run','journal','service-0','service-1'))
    print(json.dumps(result,indent=2))
