"""HF/macro target parity retaining the selected acoustic implementation and KV."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from tools.reports import context_transfer_parity_operations as original
from tools.reports import prepare_context_transfer_parity as preparation
from tools.reports.context_acoustic_tail_operations import parent_validator, read
from tools.reports.prepare_context_dual_replay import relocate
from tools.reports.prepare_context_macro_replay import write
from tools.reports.watch_context_acoustic_training import bind


def command(c, role):
    argv=original.command(c,role)
    if role.startswith('service-'):
        old='tools.reports.context_dual_runtime_candidate'
        if argv.count(old)!=1:raise ValueError('unique target acoustic service entry required')
        argv[argv.index(old)]='tools.reports.context_acoustic_service_candidate'
    return argv


_base_validate=bind(original.validate,command=command)
audit_outputs=original.audit_outputs


def validate(c):
    receipt=_base_validate(c)
    parent=c['efficiency_parent'];parent_validator(c['parent_kind'])(parent)
    if receipt['parent_manifest_sha256']!=parent['manifest_sha256']:
        raise ValueError('exact efficiency parent required')
    for key in ('host','lease','model','driver_root'):
        if c[key]!=parent[key]:raise ValueError('parent infrastructure differs: '+key)
    source=Path(parent['prepared'])/'package';package=Path(c['prepared'])/'package'
    before={str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}
    after={str(p.relative_to(package)) for p in package.rglob('*') if p.is_file()}
    mutable={'FILES.json','RUNTIME-SOURCES.json','training.json','service-0.json','service-1.json'}
    expected=(before-{'training.json'})|set(preparation.ADDITIONS)|{'evaluation.json'}
    if {n for n in after if not n.startswith('parity-inputs/')}!=expected:
        raise ValueError('only target parity additions allowed')
    for name in before-mutable-set(preparation.ADDITIONS):
        if file_hash(source/name)!=file_hash(package/name):raise ValueError('inherited runtime or asset differs: '+name)
    for name in preparation.ADDITIONS:
        if file_hash(package/name)!=file_hash(Path(c['project'])/name):raise ValueError('target implementation differs: '+name)
    identity=read(package/'RUNTIME-SOURCES.json')
    if identity!={n:file_hash(package/n) for n in after if n.endswith('.py')}:
        raise ValueError('complete runtime identity required')
    e=read(package/'evaluation.json')
    for i in range(2):
        name=f'service-{i}.json'
        expected=relocate(read(source/name),parent['remote_root'],c['remote_root'])
        actual=read(package/name)
        expected.update(pins=e['pins'],dual_runtime_identity=e['macro_client']['runtime_identity'],
            socket_path=e['macro_client']['replicas'][i]['socket'],exchange=e['macro_client']['replicas'][i]['exchange'])
        expected['policy']['max_decisions']=actual['policy']['max_decisions'] # base validator binds target width
        if expected!=actual:raise ValueError('target service must preserve parent policy and KV capacity')
    return receipt


def prepare(c, provenance):
    fn=bind(preparation.prepare,validate_parent=parent_validator(c['parent_kind']),validate=_base_validate)
    result=fn(c,provenance)
    path=Path(c['output'])/'operations.json';ops=read(path)
    ops.update(efficiency_parent=read(c['parent_operations']),parent_kind=c['parent_kind'],project=c['project'])
    validate(ops);write(path,ops)
    return result


def run(c,provenance):
    return original.lifecycle(c,provenance,validator=validate,command_builder=command)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['prepare','validate','run','collect'],required=True)
    args=p.parse_args();c,provenance=load_config(args.config,args.local)
    if args.action=='prepare':result=prepare(c,provenance)
    elif args.action=='validate':result=validate(c)
    elif args.action=='run':result=run(c,provenance)
    else:
        from tools.reports.audit_context_macro_probe import collect_and_audit
        result=collect_and_audit(c,validator=validate,command_builder=command,
            output_auditor=audit_outputs,result_dirs=('hf','evaluation','journal','service-0','service-1'))
    print(json.dumps(result,indent=2))
