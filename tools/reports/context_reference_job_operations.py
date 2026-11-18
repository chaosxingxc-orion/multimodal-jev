"""Full TRAIN reference lifecycle gated by actual same-parent endpoint parity.

Reuse the parity/shared container lifecycle. A self-reported parity JSON or a
different endpoint/runtime cannot authorize building the fixed reference table.
"""
import argparse
import json
from pathlib import Path
from types import FunctionType
import subprocess
import sys
import time

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.finalize_context_infonce import wait_exact
from tools.reports import context_train_parity_operations as lifecycle
from tools.reports.prepare_context_reference_job import audit as audit_package,audit_results
from tools.reports.audit_context_train_parity import audit as audit_parity


def read(p):return json.loads(Path(p).read_text())


def admission(c):
    spec=c['reference_job'];parity=c['parity_audit']
    # Reconstruct source/runtime and all numerical/transport evidence. Stored
    # status strings or a copied report are not independent admission evidence.
    reference=audit_package(spec,{'source':'reference_lifecycle_validation'})
    checked=audit_parity(parity,{'source':'reference_lifecycle_parity'})
    ps=parity['parity_package']
    if (spec['parent']!=ps['parent'] or checked['runtime_provenance_verified'] is not True
            or checked['comparison']['exact'] is not True or checked['comparison']['queries']!=256
            or checked['comparison']['decisions']!=60 or checked['actor_updates']!=0):
        raise ValueError('same independently audited endpoint/runtime and TRAIN256 parity required')
    root=Path(spec['output']);other=Path(ps['output'])
    build=read(root/'reference-build-config.json');tested=read(other/'parity-macro.json')
    for key in ('source_policy_id','source_contract','base_id','release_sha256','audio_token_id',
                'dual_runtime_identity','stage1_endpoint','expected_population'):
        if build[key]!=tested[key]:raise ValueError('reference differs from numerical admission: '+key)
    for name in ('RUNTIME-SOURCES.json','initial/endpoint-commit.json','initial/policy/MANIFEST.json'):
        if file_hash(root/name)!=file_hash(other/name):
            raise ValueError('reference source asset differs from numerical admission: '+name)
    for key in ('input_proof_sha256','input_receipts_sha256'):
        if build[key]!=ps['inputs'][key]:raise ValueError('reference input population differs from parity source')
    return dict(status='same_endpoint_reference_build_admitted',reference_package_audit=reference,
        parity_container_id=checked['container_id'],parity_manifest_sha256=ps['manifest_sha256'],
        parity_transport_sha256=checked['transport_sha256'],source_policy_id=build['source_policy_id'],
        runtime_identity=build['dual_runtime_identity'],finite_panel_only=True,
        full_population_parity=False,training_dispatch_allowed=False)


def validate(c):
    admission(c)
    spec=c['reference_job']
    # Reuse common host/tag/path/mount/preflight validation after independently
    # validating this operation's reference package instead of a parity package.
    def authenticated_parent(*args):return None
    validate_shared=FunctionType(lifecycle.validate.__code__,
        dict(lifecycle.validate.__globals__,audit_package=authenticated_parent),'validate_reference_lifecycle')
    proxy=dict(c,parity_package=spec)
    result=validate_shared(proxy)
    from tools.reports.context_release_deployment import audit as audit_release
    audit_release(lifecycle.command(c,result[0],result[1],producer=True),result[1])
    return result


def launch(c,provenance):
    if c.get('reference_dispatch_allowed') is not True:
        raise ValueError('explicit full TRAIN reference dispatch required')
    runner=FunctionType(lifecycle.launch.__code__,dict(lifecycle.launch.__globals__,validate=validate),
        'launch_admitted_reference')
    # The shared implementation's inference flag only controls its generic
    # service/client launcher. Public authorization is the explicit gate above.
    result=runner(dict(c,finite_inference_dispatch_allowed=True),provenance)
    return dict(result,status='collected_independent_reference_output_audit_required')


def audit(c,provenance):
    gate=admission(c);spec=dict(c['reference_job']);root=Path(spec['output'])
    handle=read(c['launch']);transport=read(c['transport']);m=read(root/'PREPARED.json')
    if (read(c['wait'])!=dict(container_id=handle['container_id'],exit_code=0)
            or file_hash(c['archive'])!=transport['archive_sha256']
            or handle['config_sha256']!=file_hash(root/'config.json')
            or handle['services']!={n:file_hash(root/p) for n,p in m['layout']['services'].items()}):
        raise ValueError('exact successful reference wait/config/archive required')
    spec.update({key:c[key] for key in ('deployment','launch','inspect','transport','collection','results_audit_output')})
    config=read(root/'config.json')
    spec['expected_argv']=lifecycle.command(c['deployment'],m,config,producer=True)
    result=audit_results(spec,provenance)
    return dict(result,status='reference_job_and_same_endpoint_parity_audited',numerical_admission=gate,
        full_population_parity=False,training_dispatch_allowed=False)


def wait_for_parity(c):
    """Wait on one pinned controller identity; numerical admission stays in launch."""
    prior=c.get('predecessor')
    if prior is None:return
    deployment=c['parity_audit']['deployment'];base=Path(deployment['operations_root'])
    expected={'driver':base/'driver/started.json','config':base/'finalization/config.json',
        'terminal':base/'finalization/terminal.json','audit_config':base/'finalization/audit-config.json'}
    for key,path in expected.items():
        if prior[key]!=str(path):raise ValueError('exact parity predecessor path required: '+key)
    for key in ('driver','config'):
        if file_hash(prior[key])!=prior[key+'_sha256']:
            raise ValueError('pinned parity predecessor changed: '+key)
    if read(prior['config'])!=deployment:raise ValueError('parity deployment differs from predecessor')
    if any(c[key]!=deployment[key] for key in ('host','lease')):
        raise ValueError('same host and lease required for reference successor')
    wait_exact(read(prior['driver']),prior['terminal'])
    terminal=read(prior['terminal'])
    if terminal.get('exit_code')!=0 or terminal.get('status')!='train_parity_collected_and_audited':
        raise ValueError('successful independent TRAIN parity required')
    if read(prior['audit_config'])!=c['parity_audit']:
        raise ValueError('parity audit inputs differ from exact predecessor')


def run_and_audit(c,provenance):
    if c.get('reference_dispatch_allowed') is not True:
        raise ValueError('explicit full TRAIN reference dispatch required')
    root=Path(c['operations_root'])/'finalization'
    check_paths({'reference_finalization':root});root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'config.json',c);commit_json(root/'provenance.json',provenance)
    try:
        wait_for_parity(c)
        result=launch(c,provenance)
        if result.get('status')!='collected_independent_reference_output_audit_required':
            raise ValueError('complete reference collection required')
        stem=Path(c['operations_root'])/c['tag']
        audit_config=dict(reference_job=c['reference_job'],parity_audit=c['parity_audit'],deployment=c,
            results_audit_output=str(root/'reference-output-audit.json'),
            **{key:str(stem)+suffix for key,suffix in (
                ('launch','-launch.json'),('inspect','-inspect.json'),('wait','-wait.json'),
                ('transport','-collected.transport.json'),('collection','-collected'),('archive','-collected.tar'))})
        commit_json(root/'audit-config.json',audit_config)
        proof=audit(audit_config,provenance)
        commit_json(audit_config['results_audit_output'],proof['output_audit'])
        commit_json(root/'audit.json',proof)
        commit_json(root/'terminal.json',dict(exit_code=0,status='reference_collected_and_audited',
            training_dispatched=False,restart=False))
        return proof
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False,preserve_handles=True))
        raise


def detach(template,local):
    c,_=load_config(template,local)
    if c.get('reference_dispatch_allowed') is not True:
        raise ValueError('explicit full TRAIN reference dispatch required')
    root=Path(c['operations_root'])/'driver'
    check_paths({'reference_driver':root});root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-B','-m','tools.reports.context_reference_job_operations',
        '--config',str(Path(template).resolve()),'--local',str(Path(local).resolve()),'--action','run']
    with (root/'driver.log').open('x') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=child.pid,argv=argv,started_unix=time.time(),
        process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19])
    commit_json(root/'started.json',record)
    return record


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['validate','launch','audit','run','detach'],default='validate')
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    if a.action=='detach':result=detach(a.config,a.local)
    elif a.action=='run':result=run_and_audit(c,provenance)
    elif a.action=='launch':result=launch(c,provenance)
    elif a.action=='audit':
        result=audit(c,provenance)
        commit_json(c['results_audit_output'],result['output_audit'])
        commit_json(c['audit_output'],result)
    else:
        m,config,_=validate(c)
        result=dict(status='reference_lifecycle_validated_not_executed',
            argv=lifecycle.command(c,m,config,producer=True),npu_executed=False,training_dispatch_allowed=False)
    print(json.dumps(result,indent=2))
