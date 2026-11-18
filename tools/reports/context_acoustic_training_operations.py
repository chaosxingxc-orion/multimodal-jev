"""Acoustic-only rollout candidate; actor, schedule, KV and objective stay fixed."""
import argparse
import json
from pathlib import Path
import shutil
from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from tools.reports import context_macro_training_operations as shared
from tools.reports import audit_context_macro_training as auditor
ADDITIONS = ('context_acoustic_batch_candidate.py', 'context_acoustic_worker_candidate.py',
             'context_acoustic_owner_candidate.py', 'context_acoustic_service_candidate.py')
from tools.reports.prepare_context_dual_replay import relocate
from tools.reports.launch_context_batch_probe_candidate import write_new
from tools.reports.run_context_batch_probe_candidate import verify


def command(c,role):
    argv=shared.command(c,role)
    if role.startswith('service-'):
        old='tools.reports.context_dual_runtime_candidate'
        if argv.count(old)!=1:raise ValueError('unique service module required')
        argv[argv.index(old)]='tools.reports.context_acoustic_service_candidate'
    return argv


def validate(c):
    receipt=shared.validate(c,command_builder=command)
    package=Path(c['prepared'])/'package';t=json.loads((package/'training.json').read_text())
    parent=c['parent_operations'];shared.validate(parent)
    remote=Path(c['remote_root']);old=Path(parent['remote_root'])
    if remote.is_relative_to(old) or old.is_relative_to(remote):
        raise ValueError('independent acoustic output required')
    source=Path(parent['prepared'])/'package'
    expected=relocate(json.loads((source/'training.json').read_text()),parent['remote_root'],c['remote_root'])
    actual=dict(t)
    for obj in (expected,actual):
        obj.pop('pins');obj['dual_runtime_identity'].pop('source_manifest_sha256')
        obj['macro_client']['runtime_identity'].pop('source_manifest_sha256',None)
    if actual!=expected or receipt['parent_manifest_sha256']!=parent['manifest_sha256']:
        raise ValueError('matched acoustic-only training settings required')
    mutable={'FILES.json','RUNTIME-SOURCES.json','training.json','service-0.json','service-1.json'}
    for path in source.rglob('*'):
        if path.is_file() and path.name not in mutable and file_hash(path)!=file_hash(package/path.relative_to(source)):
            raise ValueError('inherited training runtime or asset changed')
    if t['start_update']!=0 or t['stop_update']!=100 or t['microbatch']!=8:
        raise ValueError('fixed micro8 initial 100-step comparison required')
    for i in range(2):
        original=relocate(json.loads((source/f'service-{i}.json').read_text()),parent['remote_root'],c['remote_root'])
        candidate=json.loads((package/f'service-{i}.json').read_text())
        for obj in (original,candidate):
            obj.pop('pins');obj['dual_runtime_identity'].pop('source_manifest_sha256')
        if original!=candidate:raise ValueError('service policy/capacity changed')
    for name in ADDITIONS:
        if file_hash(package/'tools/reports'/name)!=file_hash(Path(c['project'])/'tools/reports'/name):
            raise ValueError('declared acoustic implementation changed')
    return receipt


def audit_outputs(package,collection):
    return auditor.audit_outputs(package,collection)


def run(c,provenance):
    if c.get('finite_training_dispatch_allowed') is not True:raise ValueError('explicit finite dispatch required')
    return shared.run_services(dict(c,finite_inference_dispatch_allowed=True),provenance,
        validator=validate,command_builder=command,required_devices=(4,5,6,7))


def prepare(c,provenance):
    parent=json.loads(Path(c['parent_operations']).read_text());pr=shared.validate(parent)
    source=Path(parent['prepared'])/'package';dest=Path(c['output']);dest.mkdir(parents=True,exist_ok=False)
    package=dest/'package';shutil.copytree(source,package);(package/'FILES.json').unlink()
    for name in ADDITIONS:
        shutil.copy2(Path(c['project'])/'tools/reports'/name,package/'tools/reports'/name)
    write=lambda p,v:p.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
    write(package/'RUNTIME-SOURCES.json',{str(p.relative_to(package)):file_hash(p) for p in package.rglob('*.py')})
    root=c['remote_root'];old=parent['remote_root']
    t=relocate(json.loads((source/'training.json').read_text()),old,root)
    t['dual_runtime_identity']['source_manifest_sha256']=file_hash(package/'RUNTIME-SOURCES.json')
    t['macro_client']['runtime_identity']=t['dual_runtime_identity']
    external={p:h for p,h in t['pins'].items() if not Path(p).is_relative_to(root+'/package')}
    internal={root+'/package/'+str(p.relative_to(package)):file_hash(p) for p in package.rglob('*')
        if p.is_file() and p.name not in ('training.json','service-0.json','service-1.json')}
    t['pins']=dict(external,**internal);write(package/'training.json',t)
    for i in range(2):
        s=relocate(json.loads((source/f'service-{i}.json').read_text()),old,root)
        s['dual_runtime_identity']=t['dual_runtime_identity'];s['pins']=internal
        write(package/f'service-{i}.json',s)
    write_new(package/'FILES.json',{str(p.relative_to(package)):file_hash(p) for p in package.rglob('*') if p.is_file()})
    receipt=dict(pr,files_sha256=verify(package),remote_root=root,ipc_root=root+'/ipc',
        runtime_identity=t['dual_runtime_identity'],parent_manifest_sha256=parent['manifest_sha256'],config_provenance=provenance)
    write_new(dest/'PREPARED.json',receipt)
    ops=dict(parent,parent_operations=parent,prepared=str(dest),remote_root=root,tag=c['tag'],
        operations_root=c['operations_root'],manifest_sha256=receipt['files_sha256'],
        project=c['project'],finite_training_dispatch_allowed=False)
    validate(ops);write_new(dest/'operations.json',ops);return ops


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['prepare','validate','run','collect'],required=True);a=p.parse_args()
    c,provenance=load_config(a.config,a.local)
    if a.action=='prepare':result=prepare(c,provenance)
    elif a.action=='validate':result=validate(c)
    elif a.action=='run':result=run(c,provenance)
    else:
        from tools.reports.audit_context_macro_probe import collect_and_audit
        result=collect_and_audit(c,validator=validate,command_builder=command,output_auditor=audit_outputs,
                                result_dirs=('run','journal','service-0','service-1'))
    print(json.dumps(result,indent=2))
