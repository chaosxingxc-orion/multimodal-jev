"""Two-update DP2 cached-reference acceptance on the shared four-card lifecycle.

Reconstruct actual endpoint parity and full reference evidence before launch.
This entry never admits formal training, online reference fill, or a new scope.
"""
import argparse
import inspect
import json
from pathlib import Path
from types import FunctionType

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports import context_train_parity_operations as lifecycle
from tools.reports.context_cached_storage_operations import command,storage_preflight
from tools.reports.context_reference_job_operations import audit as audit_reference
from tools.reports.prepare_context_cached_training import audit as audit_package
from tools.reports.audit_context_cached_training import audit as audit_training
from multimodal_jev.context_cached_tail_window import contract as tail_contract


def read(p):return json.loads(Path(p).read_text())


def admission(c):
    spec=c['training_package'];reference=c['reference_audit']
    package=audit_package(spec,{'source':'cached_training_lifecycle'})
    proof=audit_reference(reference,{'source':'cached_training_reference_replay'})
    job=spec['reference_job'];source=reference['reference_job']
    if spec['parent']!=source['parent']:
        raise ValueError('training must retain the numerically admitted reference parent')
    for key in ('output','manifest_sha256','remote_root'):
        if job[key]!=source[key]:raise ValueError('training reference producer differs: '+key)
    if job['collection']!=reference['collection']:
        raise ValueError('training must consume the independently collected reference')
    root=Path(spec['output']);config=read(root/'config.json')
    dataset=config['reference_dataset'];result=proof['output_audit'];gate=proof['numerical_admission']
    if (proof['status']!='reference_job_and_same_endpoint_parity_audited'
            or result['manifest_sha256']!=dataset['manifest_sha256'] or result['recordings']!=40513
            or result['policy_id']!=config['source_policy_id']
            or gate['source_policy_id']!=config['source_policy_id']
            or gate['runtime_identity']!=config['dual_runtime_identity']
            or package['actor_updates']!=2 or config['start_update']!=0 or config['stop_update']!=2
            or config['experiment_kind'] not in ('engineering_prefix','cached_reference_tail_admission')
            or config['engineering_prefix'] is not True
            or config['training_dispatch_allowed'] is not False or config['checkpoint_interval']!=1
            or config.get('resume_root') or 'reference_cache' in config):
        raise ValueError('same endpoint/cache/runtime and exactly two new updates required')
    if config['experiment_kind']=='cached_reference_tail_admission':tail_contract(config)
    parent=Path(source['output'])
    for name in ('RUNTIME-SOURCES.json','initial/endpoint-commit.json','initial/policy/MANIFEST.json'):
        if file_hash(root/name)!=file_hash(parent/name):
            raise ValueError('training source differs from admitted reference: '+name)
    return dict(status='cached_two_update_prerequisites_audited',package_audit=package,
        reference_manifest_sha256=dataset['manifest_sha256'],parity_container_id=gate['parity_container_id'],
        source_policy_id=config['source_policy_id'],actor_updates=2,
        acceptance_kind=config['experiment_kind'],training_dispatch_allowed=False)


def consumer_command(c,m,config,**_):return command(c,m,config)


def consumer_preflight(config,remote,**_):return storage_preflight(config,remote)


def validate(c):
    admission(c)
    validator=FunctionType(lifecycle.validate.__code__,dict(lifecycle.validate.__globals__,
        audit_package=lambda *a:None,command=consumer_command,storage_preflight=consumer_preflight),
        'validate_cached_training')
    result=validator(dict(c,parity_package=c['training_package']))
    from tools.reports.context_release_deployment import audit as audit_release
    audit_release(command(c,result[0],result[1]),result[1])
    return result


def launch(c,provenance):
    if c.get('finite_training_dispatch_allowed') is not True:
        raise ValueError('explicit two-update training dispatch required')
    # Adapt only the legacy writable-cache preflight. All shared locking,
    # image/device checks, exact handle/wait and collection remain unchanged.
    shared=lifecycle.shared;source=inspect.getsource(shared.launch)
    begin="        check = storage + '\\ncheck_paths('"
    end="        checked(['python3', '-c', check])"
    if source.count(begin)!=1 or source.count(end)!=1:
        raise ValueError('shared storage preflight anchor changed')
    left=source.index(begin);right=source.index(end,left)
    source=source[:left]+"        check = storage + '\\n' + storage_preflight(config,root)\n"+source[right:]
    namespace=dict(shared.launch.__globals__,validate=validate,command=command,storage_preflight=storage_preflight)
    exec(compile(source,'<shared-cached-training-lifecycle>','exec'),namespace)
    result=namespace['launch'](c,provenance)
    return dict(result,status='collected_independent_cached_training_audit_required')


def audit(c,provenance):
    prerequisites=admission(c)
    result=audit_training(c,provenance)
    return dict(result,prerequisites=prerequisites,training_dispatch_allowed=False,
        formal_training_admitted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['validate','launch','audit'],default='validate')
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    if a.action=='launch':result=launch(c,provenance)
    elif a.action=='audit':result=audit(c,provenance);commit_json(c['output'],result)
    else:
        m,config,_=validate(c)
        result=dict(status='cached_training_lifecycle_validated_not_executed',argv=command(c,m,config),
            npu_executed=False,training_dispatch_allowed=False)
    print(json.dumps(result,indent=2))
