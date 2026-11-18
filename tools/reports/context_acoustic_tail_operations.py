"""Prepare and audit an epoch-tail diagnostic retaining acoustic/KV services."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from tools.reports import context_macro_tail_operations_v2 as tail
from tools.reports import context_macro_tail_operations as original
from tools.reports import context_acoustic_training_operations as acoustic
from tools.reports import context_kv_training_operations as kv
from tools.reports import prepare_context_macro_tail as preparation
from tools.reports.prepare_context_dual_replay import relocate
from tools.reports.prepare_context_macro_replay import write
from tools.reports.watch_context_acoustic_training import bind

ADDITIONS = (*preparation.ADDITIONS, 'tools/reports/context_macro_tail_entry_v2.py')


def read(path): return json.loads(Path(path).read_text())


def command(c, role):
    argv = tail.command(c, role)
    if role.startswith('service-'):
        old = 'tools.reports.context_dual_runtime_candidate'
        if argv.count(old) != 1: raise ValueError('unique acoustic service entry required')
        argv[argv.index(old)] = 'tools.reports.context_acoustic_service_candidate'
    return argv


_base_validate = bind(original.validate, command=command)
audit_outputs = original.audit_outputs


def parent_validator(kind):
    if kind == 'acoustic': return acoustic.validate
    if kind == 'kv16': return kv.validate
    raise ValueError('explicit acoustic or kv16 parent kind required')


def validate(c):
    receipt = _base_validate(c)
    parent = c['efficiency_parent']; parent_validator(c['parent_kind'])(parent)
    source = Path(parent['prepared'])/'package'; package = Path(c['prepared'])/'package'
    if receipt['parent_manifest_sha256'] != parent['manifest_sha256']:
        raise ValueError('exact efficiency parent required')
    for key in ('host', 'lease', 'model', 'driver_root', 'shared_root'):
        if c[key] != parent[key]: raise ValueError('parent infrastructure differs: '+key)
    before = {str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}
    after = {str(p.relative_to(package)) for p in package.rglob('*') if p.is_file()}
    if after != before | set(ADDITIONS): raise ValueError('only declared tail runtime additions allowed')
    mutable = {'FILES.json','RUNTIME-SOURCES.json','training.json','service-0.json','service-1.json'}
    for name in before - mutable:
        if file_hash(source/name) != file_hash(package/name): raise ValueError('inherited asset differs: '+name)
    for name in ADDITIONS:
        if file_hash(package/name) != file_hash(Path(c['project'])/name):
            raise ValueError('declared diagnostic source differs: '+name)
    internal = {c['remote_root']+'/package/'+name: file_hash(package/name)
                for name in after - {'FILES.json','training.json','service-0.json','service-1.json'}}
    runtime = read(package/'RUNTIME-SOURCES.json')
    if runtime != {name:file_hash(package/name) for name in after if name.endswith('.py')}:
        raise ValueError('complete runtime source identity required')
    identity = relocate(read(source/'training.json')['dual_runtime_identity'],parent['remote_root'],c['remote_root'])
    identity['source_manifest_sha256'] = file_hash(package/'RUNTIME-SOURCES.json')
    for name in ('training.json','service-0.json','service-1.json'):
        expected = relocate(read(source/name),parent['remote_root'],c['remote_root'])
        expected['dual_runtime_identity'] = identity
        if name == 'training.json':
            expected.update(experiment_kind='tail_admission',stop_update=2,engineering_prefix=True,
                            training_dispatch_allowed=False,checkpoint_interval=1)
            expected['macro_client']['runtime_identity'] = identity
            expected['macro_client']['attempt_id'] = 'tail-admission-v1'
            external = {p:h for p,h in expected['pins'].items()
                        if not Path(p).is_relative_to(c['remote_root']+'/package')}
            expected['pins'] = dict(external,**internal)
        else: expected['pins'] = internal
        if read(package/name) != expected: raise ValueError('tail must preserve parent policy/capacity/settings: '+name)
    return receipt


def prepare(c, provenance):
    validator = parent_validator(c['parent_kind'])
    fn = bind(preparation.prepare, validate_parent=validator, validate=_base_validate, ADDITIONS=ADDITIONS)
    result = fn(c, provenance)
    path = Path(c['output'])/'operations.json'; ops = read(path)
    ops.update(efficiency_parent=read(c['parent_operations']),parent_kind=c['parent_kind'],project=c['project'])
    validate(ops); write(path,ops)
    return result


run = bind(original.run, validate=validate, command=command)
audit_collected = bind(original.audit_collected, validate=validate, command=command)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--local', required=True)
    p.add_argument('--action', choices=['prepare','validate','run','collect','audit'],required=True)
    args = p.parse_args(); c, provenance = load_config(args.config,args.local)
    if args.action=='prepare': result=prepare(c,provenance)
    elif args.action=='validate': result=validate(c)
    elif args.action=='run': result=run(c,provenance)
    elif args.action=='audit': result=audit_collected(c)
    else:
        from tools.reports.audit_context_macro_probe import collect_and_audit
        result=collect_and_audit(c,validator=validate,command_builder=command,
            output_auditor=audit_outputs,result_dirs=('run','journal','service-0','service-1'))
    print(json.dumps(result,indent=2))
