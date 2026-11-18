"""Fixed 100-update cached GRPO package, lifecycle and independent audit.

Reuses the sealed two-update trainer unchanged. Dispatch requires independent
prefix, tail and fresh-process recovery evidence for that exact runtime.
"""
import argparse
import inspect
import json
from pathlib import Path
from types import FunctionType

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports import prepare_context_cached_training as package
from tools.reports import context_cached_training_operations as lifecycle
from tools.reports import context_cached_recovery_operations as recovery
from tools.reports import audit_context_cached_training as outputs
from tools.reports.context_macro_trainer_candidate import experiment_window
from tools.reports.context_macro_efficiency_summary import summarize


def read(path):return json.loads(Path(path).read_text())


def gates(c,provenance):
    evidence=c['acceptance']
    if set(evidence)!={'prefix','tail','recovery'}:
        raise ValueError('prefix, tail and recovery evidence required')
    proofs={n:(recovery if n=='recovery' else lifecycle).audit(evidence[n],provenance)
            for n in ('prefix','tail','recovery')}
    expected={'prefix':'finite_cached_reference_training_outputs_audited',
        'tail':'finite_cached_reference_tail_outputs_audited','recovery':'finite_cached_reference_recovery_audited'}
    if any(proofs[n]['status']!=expected[n] for n in expected):
        raise ValueError('successful independently audited finite gates required')
    prefix=evidence['prefix']['training_package'];tail=evidence['tail']['training_package']
    if (evidence['recovery']['training_package']['parent_audit']!=evidence['prefix']
            or prefix.get('acceptance_kind','engineering_prefix')!='engineering_prefix'
            or tail.get('acceptance_kind')!='cached_reference_tail_admission'
            or proofs['recovery'].get('fresh_process_npu_resume_verified') is not True
            or c['parent']!=prefix['parent'] or c['reference_job']!=prefix['reference_job']
            or tail['parent']!=prefix['parent'] or tail['reference_job']!=prefix['reference_job']
            or evidence['tail']['reference_audit']!=evidence['prefix']['reference_audit']):
        raise ValueError('same source/runtime/reference and continuous replay parent required')
    root=Path(prefix['output']);config=read(root/'config.json')
    if (config['global_batch']!=128 or config['microbatch']!=8
            or config['batch_config']['training_devices']!=2
            or config['batch_config']['inference_instances']!=2
            or config['batch_config']['inference_batch_queries']!=32):
        raise ValueError('admitted DP2 micro8 service32 baseline required')
    return proofs


def configured(original,old,remote,reference,kind='efficiency'):
    if kind!='efficiency':raise ValueError('explicit efficiency kind required')
    c=package.configured(original,old,remote,reference)
    c.update(stop_update=100,experiment_kind='efficiency',checkpoint_interval=1)
    experiment_window(c)
    return c


def runner(manifest,kind='efficiency'):
    if kind!='efficiency':raise ValueError('explicit efficiency kind required')
    source=package.runner(manifest)
    before="if c['start_update']!=0 or c['stop_update']!=2:"
    after=("if (c['start_update']!=0 or c['stop_update']!=100 or c.get('resume_root')\n"
        " or c['experiment_kind']!='efficiency' or c['checkpoint_interval']!=1):")
    if source.count(before)!=1:raise ValueError('shared finite launch guard changed')
    source=source.replace(before,after);compile(source,'<cached-efficiency-runner>','exec')
    return source


def prepared_inputs(c,provenance):
    gates(c,provenance)
    return package.inputs(c,provenance)


def package_action(c,provenance,*,audit=False):
    if c.get('acceptance_kind')!='efficiency':raise ValueError('efficiency package declaration required')
    fn=package.audit if audit else package.prepare
    action=FunctionType(fn.__code__,dict(fn.__globals__,inputs=prepared_inputs,
        configured=configured,runner=runner),'cached_efficiency_package')
    result=action(c,provenance)
    if audit:
        manifest=read(Path(c['output'])/'PREPARED.json')
        if manifest['experiment']!=dict(kind='efficiency',updates=100,warmup_updates=5,checkpoint_interval=1):
            raise ValueError('explicit 100-step efficiency manifest required')
        result.update(status='cached_efficiency_package_audited',actor_updates=100)
    else:
        path=Path(c['output'])/'PREPARED.json';manifest=read(path)
        manifest['experiment']=dict(kind='efficiency',updates=100,warmup_updates=5,checkpoint_interval=1)
        # This is still the unpublished prepare transaction. No run consumes
        # this candidate until its final manifest hash has been returned.
        path.write_text(json.dumps(manifest,indent=2)+'\n')
        result['manifest_sha256']=file_hash(path)
    return result


def validate(c):
    spec=c['training_package'];package_action(spec,{'source':'efficiency_lifecycle'},audit=True)
    prefix=spec['acceptance']['prefix']
    if c['reference_audit']!=prefix['reference_audit']:
        raise ValueError('exact independently admitted reference required')
    if any(c[k]!=prefix['deployment'][k] for k in ('host','lease')):
        raise ValueError('same host and exclusive resource lease required')
    validator=FunctionType(lifecycle.validate.__code__,dict(lifecycle.validate.__globals__,admission=lambda _:None))
    return validator(c)


def launch(c,provenance):
    if c.get('efficiency_dispatch_allowed') is not True:
        raise ValueError('explicit 100-step efficiency authorization required')
    fn=FunctionType(lifecycle.launch.__code__,dict(lifecycle.launch.__globals__,validate=validate))
    return fn(dict(c,finite_training_dispatch_allowed=True),provenance)


def efficiency_shared_auditor(namespace):
    source=inspect.getsource(outputs.shared.audit_outputs)
    before="if experiment_window(c) != 'engineering_prefix' or start != 0:"
    after="if experiment_window(c) != 'efficiency' or start != 0 or stop != 100 or c['checkpoint_interval'] != 1:"
    if source.count(before)!=1:raise ValueError('shared output window guard changed')
    source=source.replace(before,after)
    scope=dict(namespace);exec(compile(source,'<cached-efficiency-output-audit>','exec'),scope)
    return scope['audit_outputs']


def audit_outputs(root,collection,*,reference_root):
    root,collection=Path(root),Path(collection);c=read(root/'config.json')
    if (experiment_window(c)!='efficiency' or c['stop_update']!=100 or c['checkpoint_interval']!=1):
        raise ValueError('complete fresh 100-update efficiency interval required')
    verifier=outputs.ReferenceConsumption(c['reference_dataset'],reference_root);receipts={}
    for index in range(100):
        receipt=read(collection/f'run/rollout-{index:04d}.json')
        if receipt['update']!=index:raise ValueError('exact ordered rollout cursor required')
        verifier.check(receipt,read(collection/f'run/batch-{index:04d}.json'));receipts[index]=receipt
    def transition(receipt,seen,*,current,reference,replicas):
        if (receipt!=receipts.get(receipt['update']) or reference!=verifier.policy
                or receipt['behavior_policy_id']!=current or replicas!=2):
            raise ValueError('authenticated cached efficiency transition required')
        return []
    scope=dict(outputs.shared.audit_outputs.__globals__,expected_contract=outputs.expected_contract,
        reference_transition=transition)
    proof=efficiency_shared_auditor(scope)(root,collection,initial_policy=root/'initial/policy')
    stats=summarize(collection/'run',start_update=0,stop_update=100,warmup_updates=5)
    if verifier.samples!=12800:raise ValueError('12,800 real query exposures required')
    return dict(proof,status='cached_100step_efficiency_outputs_audited',experiment_kind='efficiency',
        consumed_reference_queries=verifier.samples,online_reference_calls=0,reference_policy_switches=0,
        statistics=stats,formal_training_admitted=False,training_dispatch_allowed=False)


def audit(c,provenance):
    # The transport implementation remains identical; only the independently
    # reconstructed package and the registered output window differ.
    fn=FunctionType(outputs.audit.__code__,dict(outputs.audit.__globals__,
        audit_package=lambda spec,p:package_action(spec,p,audit=True),audit_outputs=audit_outputs))
    if c['reference_audit']!=c['training_package']['acceptance']['prefix']['reference_audit']:
        raise ValueError('efficiency reference evidence changed')
    return fn(c,provenance)


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['prepare','audit-package','validate','launch','audit'],default='validate')
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    if a.action=='prepare':result=package_action(c,provenance)
    elif a.action=='audit-package':result=package_action(c,provenance,audit=True)
    elif a.action=='launch':result=launch(c,provenance)
    elif a.action=='audit':result=audit(c,provenance);commit_json(c['output'],result)
    else:validate(c);result=dict(status='efficiency_validated_only',training_dispatched=False)
    print(json.dumps(result,indent=2))
