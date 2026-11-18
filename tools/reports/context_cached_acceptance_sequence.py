"""Run prefix, real-tail and fresh-process cached GRPO gates after reference.

Fixed finite sequence only; never dispatch formal training, refresh references,
retry a failed attempt, or invoke a model to drive routine operations.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.finalize_context_infonce import wait_exact
from tools.reports import context_reference_job_operations as reference
from tools.reports import context_cached_training_operations as training
from tools.reports import context_cached_recovery_operations as recovery
from tools.reports import prepare_context_cached_training as prepare_training
from tools.reports import prepare_context_cached_recovery as prepare_recovery

STAGES=('prefix','tail','recovery')
STATUSES={'prefix':'finite_cached_reference_training_outputs_audited',
          'tail':'finite_cached_reference_tail_outputs_audited',
          'recovery':'finite_cached_reference_recovery_audited'}


def read(path):return json.loads(Path(path).read_text())


def validate(c):
    if c.get('finite_training_dispatch_allowed') is not True:
        raise ValueError('explicit finite cached acceptance authorization required')
    if set(c['stages'])!=set(STAGES):raise ValueError('exact prefix, tail and recovery sequence required')
    prior=c['reference_predecessor'];ops=read(prior['config'])
    base=Path(ops['operations_root'])
    for key,path in {'driver':base/'driver/started.json','config':base/'finalization/config.json',
                     'terminal':base/'finalization/terminal.json',
                     'audit_config':base/'finalization/audit-config.json'}.items():
        if prior[key]!=str(path):raise ValueError('exact reference predecessor path required: '+key)
    for key in ('driver','config'):
        if file_hash(prior[key])!=prior[key+'_sha256']:
            raise ValueError('reference predecessor changed: '+key)
    if ops.get('reference_dispatch_allowed') is not True:
        raise ValueError('authorized reference predecessor required')
    # All generated packages are isolated. Reusing a previous directory is never
    # interpreted as permission to resume or retry an incomplete operation.
    local_paths=[Path(c['output'])]
    remote_paths=[]
    tags=[]
    for name in STAGES:
        stage=c['stages'][name]
        if set(stage)!={'package','remote_root','operations_root','tag'}:
            raise ValueError('stage only declares locations and tag')
        for key in ('package','operations_root','remote_root'):
            training.lifecycle.shared.path(stage[key])
        local_paths.extend([Path(stage['package']),Path(stage['operations_root'])])
        remote_paths.append(Path(stage['remote_root']));tags.append(stage['tag'])
    import re
    if len(set(tags))!=3 or any(not re.fullmatch('[a-z0-9][a-z0-9-]*',t) for t in tags):
        raise ValueError('distinct safe acceptance tags required')
    protected_local=[base,Path(ops['package']),Path(ops['reference_job']['parent']['output'])]
    protected_remote=[Path(ops['reference_job']['remote_root']),
                      Path(ops['reference_job']['parent']['remote_root'])]
    for paths,protected in ((local_paths,protected_local),(remote_paths,protected_remote)):
        paths=[p.resolve() for p in paths];protected=[p.resolve() for p in protected]
        for i,p in enumerate(paths):
            if any(p.is_relative_to(q) or q.is_relative_to(p) for q in paths[:i]+protected):
                raise ValueError('isolated acceptance paths required')
    check_paths({'acceptance':c['output'],**{str(i):p for i,p in enumerate(local_paths[1:])}})
    return ops


def wait_reference(c):
    ops=validate(c);prior=c['reference_predecessor']
    wait_exact(read(prior['driver']),prior['terminal'])
    terminal=read(prior['terminal'])
    if terminal.get('exit_code')!=0 or terminal.get('status')!='reference_collected_and_audited':
        raise ValueError('successful independent reference predecessor required')
    spec=read(prior['audit_config'])
    if (spec['deployment']!=ops or spec['reference_job']!=ops['reference_job']
            or spec['parity_audit']!=ops['parity_audit']):
        raise ValueError('reference audit differs from pinned predecessor')
    # launch and training admission subsequently reconstruct the numerical,
    # transport and full-population proofs, not merely these status strings.
    proof=reference.audit(spec,{'source':'cached_acceptance_reference_gate'})
    if proof['status']!='reference_job_and_same_endpoint_parity_audited':
        raise ValueError('independent reference audit required')
    return ops,spec,proof


def producer_spec(audit_config):
    spec=dict(audit_config['reference_job'])
    for key in ('deployment','launch','inspect','transport','collection','results_audit_output'):
        spec[key]=audit_config[key]
    root=Path(spec['output'])
    spec['expected_argv']=reference.lifecycle.command(spec['deployment'],read(root/'PREPARED.json'),
        read(root/'config.json'),producer=True)
    return spec


def audit_config(operations,reference_audit):
    stem=Path(operations['operations_root'])/operations['tag']
    return dict(training_package=operations['training_package'],reference_audit=reference_audit,
        deployment=operations,**{key:str(stem)+suffix for key,suffix in (
            ('launch','-launch.json'),('inspect','-inspect.json'),('wait','-wait.json'),
            ('transport','-collected.transport.json'),('collection','-collected'),('archive','-collected.tar'))})


def run_stage(name,stage,reference_ops,reference_audit,prefix_audit,provenance):
    if name not in STAGES:raise ValueError('unknown finite acceptance stage')
    if name=='recovery':
        if prefix_audit is None:raise ValueError('audited continuous prefix required')
        spec=dict(parent_audit=prefix_audit,output=stage['package'],remote_root=stage['remote_root'])
        preparer,module=prepare_recovery,recovery
    else:
        spec=dict(parent=reference_ops['reference_job']['parent'],reference_job=producer_spec(reference_audit),
            output=stage['package'],remote_root=stage['remote_root'],
            acceptance_kind='engineering_prefix' if name=='prefix' else 'cached_reference_tail_admission')
        preparer,module=prepare_training,training
    root=Path(stage['operations_root'])/'finalization'
    check_paths({'finite_finalization':root});root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'preparation-config.json',spec)
    try:
        prepared=preparer.prepare(spec,provenance)
        spec=dict(spec,manifest_sha256=prepared['manifest_sha256'])
        from tools.reports.context_release_deployment import inherited
        ops=inherited(reference_ops)
        ops.update(training_package=spec,reference_audit=reference_audit,package=stage['package'],
            manifest_sha256=spec['manifest_sha256'],operations_root=stage['operations_root'],tag=stage['tag'],
            finite_training_dispatch_allowed=True)
        commit_json(root/'config.json',ops);commit_json(root/'provenance.json',provenance)
        collected=module.launch(ops,provenance)
        if collected.get('status')!='collected_independent_cached_training_audit_required':
            raise ValueError('complete finite training collection required')
        bound=audit_config(ops,reference_audit);commit_json(root/'audit-config.json',bound)
        proof=module.audit(bound,provenance)
        if proof.get('status')!=STATUSES[name] or proof.get('formal_training_admitted') is not False:
            raise ValueError('independent finite acceptance proof required')
        commit_json(root/'audit.json',proof)
        commit_json(root/'terminal.json',dict(exit_code=0,status=STATUSES[name],formal_training_dispatched=False))
        return bound,proof
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False,preserve_handles=True))
        raise


def run(c,provenance):
    validate(c);root=Path(c['output']);root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'config.json',c);commit_json(root/'provenance.json',provenance)
    try:
        commit_json(root/'waiting.json',c['reference_predecessor'])
        ops,ref,proof=wait_reference(c);commit_json(root/'reference-audit.json',proof)
        prefix=None;results={}
        for name in STAGES:
            bound,proof=run_stage(name,c['stages'][name],ops,ref,prefix,provenance)
            if name=='prefix':prefix=bound
            results[name]=proof
            commit_json(root/(name+'-result.json'),proof)
        commit_json(root/'terminal.json',dict(exit_code=0,status='cached_finite_sequence_audited',
            stages=list(STAGES),formal_training_dispatched=False,efficiency_and_effects_pending=True))
        return results
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False,preserve_handles=True))
        raise


def detach(template,local):
    c,_=load_config(template,local);validate(c)
    root=Path(c['output']).with_name(Path(c['output']).name+'-driver')
    check_paths({'acceptance_driver':root});root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-B','-m','tools.reports.context_cached_acceptance_sequence',
        '--config',str(Path(template).resolve()),'--local',str(Path(local).resolve()),'--action','run']
    with (root/'driver.log').open('x') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=child.pid,argv=argv,started_unix=time.time(),
        process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19])
    commit_json(root/'started.json',record);return record


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['validate','run','detach'],default='validate');a=p.parse_args()
    c,provenance=load_config(a.config,a.local)
    if a.action=='detach':result=detach(a.config,a.local)
    elif a.action=='run':result=run(c,provenance)
    else:validate(c);result=dict(status='finite_sequence_validated_only',npu_executed=False)
    print(json.dumps(result,indent=2))
