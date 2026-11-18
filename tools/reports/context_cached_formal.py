"""Full clean-SLURP cached GRPO package and independent output audit.

Preparation reuses the admitted runtime bytewise. Dispatch requires the same
runtime's finite and 100-step efficiency evidence plus an explicit formal flag.
"""
import argparse
import inspect
import json
from pathlib import Path
from types import FunctionType

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_topology_batch_plan import derive
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports import context_cached_efficiency as efficiency
from tools.reports import prepare_context_cached_training as package
from tools.reports import audit_context_cached_training as outputs
from tools.reports import context_cached_training_operations as lifecycle
from tools.reports.context_replica_formal_candidate import contract
from tools.reports.prepare_context_replica_training_candidate import ENGINEERING_GATE,FORMAL_GATE

STEPS=[0,*range(64,634,64),634]


def read(path):return json.loads(Path(path).read_text())


def declared_curve(c):
    if c.get('evaluation_steps')!=STEPS or any(type(n) is not int for n in c['evaluation_steps']):
        raise ValueError('registered 11-point DP2 curve through update634 required')
    return list(STEPS)


def gates(c,provenance):
    declared_curve(c)
    proof=efficiency.audit(c['efficiency_audit'],provenance)
    if (proof['status']!='cached_100step_efficiency_outputs_audited'
            or proof['consumed_reference_queries']!=12800 or proof['online_reference_calls']!=0):
        raise ValueError('complete independently audited 100-step efficiency run required')
    prior=c['efficiency_audit']['training_package']
    if (c['parent']!=prior['parent'] or c['reference_job']!=prior['reference_job']):
        raise ValueError('formal initialization must be original Stage1 endpoint and reference')
    return proof


def configured(original,old,remote,reference,kind='formal_source',*,steps):
    if kind!='formal_source':raise ValueError('explicit formal source kind required')
    declared_curve(dict(evaluation_steps=steps))
    config=package.configured(original,old,remote,reference)
    plan=derive(config['batch_config'])
    if (plan['dp_replicas']!=2 or plan['global_batch']!=128 or plan['total_updates']!=634
            or plan['configuration']['source_queries']!=40513 or config['microbatch']!=8
            or plan['configuration']['inference_instances']!=2
            or plan['configuration']['inference_batch_queries']!=32):
        raise ValueError('full two-epoch DP2 micro8 service32 baseline required')
    config.update(experiment_kind='formal_source',engineering_prefix=False,
        training_dispatch_allowed=True,stop_update=plan['total_updates'],evaluation_steps=list(steps))
    config.pop('checkpoint_interval',None)
    contract(config)
    return config


def runner(manifest,kind='formal_source'):
    if kind!='formal_source':raise ValueError('explicit formal source kind required')
    source=package.runner(manifest)
    if source.count(ENGINEERING_GATE)!=1:raise ValueError('shared runner gate changed')
    source=source.replace(ENGINEERING_GATE,FORMAL_GATE)
    compile(source,'<cached-formal-runner>','exec');return source


def package_action(c,provenance,*,audit=False):
    if c.get('acceptance_kind')!='formal_source':raise ValueError('formal package declaration required')
    steps=declared_curve(c)
    def inputs(spec,p):
        gates(spec,p)
        return package.inputs(spec,p)
    def configure(original,old,remote,reference,kind):
        return configured(original,old,remote,reference,kind,steps=steps)
    fn=package.audit if audit else package.prepare
    action=FunctionType(fn.__code__,dict(fn.__globals__,inputs=inputs,configured=configure,runner=runner))
    result=action(c,provenance)
    path=Path(c['output'])/'PREPARED.json';manifest=read(path)
    experiment=contract(read(Path(c['output'])/'config.json'))
    if audit:
        if manifest.get('experiment')!=experiment:raise ValueError('formal experiment manifest differs')
        return dict(result,status='cached_formal_package_audited_not_dispatched',actor_updates=634,
            formal_dispatch_admitted=False)
    manifest['experiment']=experiment
    # Finalize the unpublished prepare transaction before exposing its digest.
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    return dict(result,manifest_sha256=file_hash(path),formal_dispatch_admitted=False)


def expected_contract(config,mounted):
    result=outputs.expected_contract(config,mounted)
    result['formal_experiment']=contract(config)
    return result


def formal_shared_auditor(namespace):
    # Keep queued finite auditors unchanged. Only this formal caller corrects
    # the formal-only raw-config argument to the derived schedule validator.
    source=inspect.getsource(outputs.shared.audit_outputs)
    before="validate_schedule(batches, population.keys(), c['batch_config'])"
    if source.count(before)!=1:raise ValueError('shared formal schedule anchor changed')
    source=source.replace(before,'validate_schedule(batches, population.keys(), plan)')
    scope=dict(namespace);exec(compile(source,'<cached-formal-output-audit>','exec'),scope)
    return scope['audit_outputs']


def audit_outputs(root,collection,*,reference_root):
    root,collection=Path(root),Path(collection);config=read(root/'config.json')
    formal=contract(config);declared_curve(config)
    verifier=outputs.ReferenceConsumption(config['reference_dataset'],reference_root)
    receipts={}
    for index in range(formal['updates']):
        receipt=read(collection/f'run/rollout-{index:04d}.json')
        batch=read(collection/f'run/batch-{index:04d}.json')
        if receipt['update']!=index:raise ValueError('ordered formal rollout cursor required')
        verifier.check(receipt,batch);receipts[index]=receipt
    def transition(receipt,seen,*,current,reference,replicas):
        if (receipt!=receipts.get(receipt['update']) or reference!=verifier.policy
                or receipt['behavior_policy_id']!=current or replicas!=2):
            raise ValueError('authenticated formal cached transition required')
        return []
    scope=dict(outputs.shared.audit_outputs.__globals__,expected_contract=expected_contract,
        reference_transition=transition)
    proof=formal_shared_auditor(scope)(root,collection,initial_policy=root/'initial/policy')
    real=sum(row['real_queries'] for row in proof['updates'])
    if (proof['status']!='formal_replica_training_audited' or real!=81026
            or len(proof['updates'])!=634 or verifier.samples!=634*128):
        raise ValueError('complete 634 updates, 81,026 real exposures and exact padded slots required')
    return dict(proof,status='formal_cached_reference_training_outputs_audited',
        real_query_exposures=real,consumed_reference_slots=verifier.samples,
        padding_slots=verifier.samples-real,online_reference_calls=0,reference_policy_switches=0,
        reference_manifest_sha256=config['reference_dataset']['manifest_sha256'],
        efficacy_evaluated=False,training_dispatch_allowed=False)


def audit(c,provenance):
    if c['reference_audit']!=c['training_package']['efficiency_audit']['reference_audit']:
        raise ValueError('same independently admitted reference required')
    fn=FunctionType(outputs.audit.__code__,dict(outputs.audit.__globals__,
        audit_package=lambda spec,p:package_action(spec,p,audit=True),audit_outputs=audit_outputs))
    return fn(c,provenance)


def validate(c):
    spec=c['training_package'];package_action(spec,{'source':'formal_lifecycle'},audit=True)
    prior=spec['efficiency_audit']
    if (c['reference_audit']!=prior['reference_audit']
            or any(c[k]!=prior['deployment'][k] for k in ('host','lease'))):
        raise ValueError('same audited reference, host and exclusive resource lease required')
    validator=FunctionType(lifecycle.validate.__code__,dict(lifecycle.validate.__globals__,admission=lambda _:None))
    return validator(c)


def launch(c,provenance):
    if c.get('formal_training_dispatch_allowed') is not True:
        raise ValueError('explicit full source training authorization required')
    fn=FunctionType(lifecycle.launch.__code__,dict(lifecycle.launch.__globals__,validate=validate))
    return fn(dict(c,finite_training_dispatch_allowed=True),provenance)


def run(c,provenance):
    """One exact launch/collect/audit transaction; failures never auto-retry."""
    if c.get('formal_training_dispatch_allowed') is not True:
        raise ValueError('explicit full source training authorization required')
    root=Path(c['operations_root'])/'finalization'
    check_paths({'formal_finalization':root});root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'config.json',c);commit_json(root/'provenance.json',provenance)
    try:
        result=launch(c,provenance)
        if result.get('status')!='collected_independent_cached_training_audit_required':
            raise ValueError('complete formal collection required')
        from tools.reports.context_cached_acceptance_sequence import audit_config
        bound=audit_config(c,c['reference_audit']);commit_json(root/'audit-config.json',bound)
        proof=audit(bound,provenance)
        if (proof.get('status')!='formal_cached_reference_training_outputs_audited'
                or proof.get('real_query_exposures')!=81026 or len(proof.get('updates',[]))!=634):
            raise ValueError('independent complete formal training audit required')
        commit_json(root/'audit.json',proof)
        commit_json(root/'terminal.json',dict(exit_code=0,status='formal_cached_training_collected_and_audited',
            selection_evaluation_pending=True,transfer_evaluation_pending=True,restart=False))
        return proof
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False,preserve_handles=True))
        raise


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['prepare','audit-package','validate','launch','audit','run'],default='audit-package')
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    if a.action=='prepare':result=package_action(c,provenance)
    elif a.action=='audit-package':result=package_action(c,provenance,audit=True)
    elif a.action=='launch':result=launch(c,provenance)
    elif a.action=='run':result=run(c,provenance)
    elif a.action=='validate':
        validate(c);result=dict(status='formal_package_lifecycle_validated_not_executed',npu_executed=False)
    else:result=audit(c,provenance);commit_json(c['output'],result)
    print(json.dumps(result,indent=2))
