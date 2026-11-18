"""Default editable-objective tail, efficiency and full-source lifecycle.

Default plugin bytes/options are bound to real prefix/recovery evidence.
Independent shared auditors replay the same default formula and all model,
reference, optimizer and transport checks with the explicit plugin contract.
"""
import argparse
import inspect
import json
from pathlib import Path
from types import FunctionType

from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json
from tools.reports import prepare_stage2_objective_run as preparation
from tools.reports import context_cached_training_operations as cached
from tools.reports import audit_context_cached_training as outputs
from tools.reports import context_cached_efficiency as efficiency
from tools.reports import context_cached_formal as formal
from tools.reports import audit_objective_training as prefix
from tools.reports import audit_objective_recovery as recovery
from tools.reports.context_objective_storage import command
from multimodal_jev.objective_plugin import ObjectivePlugin


def admission(c):
    spec=c['training_package'];preparation.audit(spec,{'source':'stage2_run_lifecycle'})
    evidence=c['acceptance'];kind=spec['kind']
    required={'prefix','recovery'}|({'tail'} if kind in ('efficiency','formal') else set())|({'efficiency'} if kind=='formal' else set())
    if set(evidence)!=required:raise ValueError('complete prerequisite set required')
    p=prefix.audit(evidence['prefix'],{'source':'stage2_prefix_admission'})
    r=recovery.audit(evidence['recovery'],{'source':'stage2_recovery_admission'})
    if (p['status']!='finite_objective_training_outputs_audited' or r['status']!='finite_objective_recovery_audited'
            or r['fresh_process_npu_resume_verified'] is not True or r['replay']['exact_objective_traces'] is not True
            or evidence['prefix']['training_package']!=spec['prefix_package']
            or evidence['recovery']['training_package']['parent_audit']!=evidence['prefix']):
        raise ValueError('same actual prefix and exact recovery required')
    config=preparation.read(Path(spec['output'])/'config.json');plugin=ObjectivePlugin(prefix.packaged_plugin(spec['output'],config))
    if plugin.identity!=p['objective_plugin'] or plugin.identity['options']!={}:
        raise ValueError('default admitted objective required')
    reference=evidence['prefix']['deployment']['reference_audit']
    if c['reference_audit']!=reference:raise ValueError('same admitted reference required')
    for key in ('host','lease'):
        if c[key]!=evidence['prefix']['deployment'][key]:raise ValueError('same four-card ownership required')
    for name in ('tail','efficiency'):
        if name not in required:continue
        prior=evidence[name]
        if prior['training_package']['prefix_package']!=spec['prefix_package'] or prior['training_package']['kind']!=name:
            raise ValueError('same prefix and correct prerequisite kind required')
        prior_ops=prior['deployment']
        if (prior_ops['reference_audit']!=reference or any(prior_ops[k]!=c[k] for k in ('host','lease'))
                or any(prior_ops['acceptance'][k]!=evidence[k] for k in ('prefix','recovery'))):
            raise ValueError('prerequisite ownership and acceptance lineage differ')
        report=audit(prior,{'source':'stage2_'+name+'_admission'})
        if report['status']!='stage2_objective_'+name+'_audited':raise ValueError('audited prerequisite required')
    return dict(status='stage2_objective_'+kind+'_admitted',objective_plugin=plugin.identity)


def audit_outputs(package,collection,*,reference_root):
    config=preparation.read(Path(package)/'config.json')
    plugin=ObjectivePlugin(prefix.packaged_plugin(package,config))
    kind={'cached_reference_tail_admission':'tail','efficiency':'efficiency','formal_source':'formal'}[config['experiment_kind']]
    original=outputs.expected_contract
    def contract(c,mounted):
        value=original(c,mounted)
        if kind=='formal':value['formal_experiment']=formal.contract(c)
        return dict(value,objective_plugin=plugin.identity)
    # Each output wrapper supplies expected_contract to the same shared auditor.
    from types import SimpleNamespace
    view=SimpleNamespace(**vars(outputs));view.expected_contract=contract
    module=outputs if kind=='tail' else efficiency if kind=='efficiency' else formal
    scope=dict(module.audit_outputs.__globals__,expected_contract=contract,outputs=view)
    fn=FunctionType(module.audit_outputs.__code__,scope)
    result=fn(package,collection,reference_root=reference_root)
    return dict(result,status='stage2_objective_'+kind+'_audited',objective_plugin=plugin.identity,
        objective_verification='exact_accepted_default_bytes_with_original_formula_replay')


def audit(c,provenance):
    spec=c['training_package']
    bound=dict(c,training_package=dict(spec,reference_job=spec['prefix_package']['parent']['reference_job']))
    fn=FunctionType(outputs.audit.__code__,dict(outputs.audit.__globals__,
        audit_package=lambda _,p:preparation.audit(spec,p),audit_outputs=audit_outputs,command=command))
    return fn(bound,provenance)


def validate(c):
    admission(c)
    from tools.reports import context_train_parity_operations as lifecycle
    fn=FunctionType(lifecycle.validate.__code__,dict(lifecycle.validate.__globals__,audit_package=lambda *_:None,
        command=lambda c,m,config,**_:command(c,m,config),storage_preflight=cached.consumer_preflight))
    result=fn(dict(c,parity_package=c['training_package']))
    from tools.reports.context_release_deployment import audit as audit_release
    audit_release(command(c,result[0],result[1]),result[1]);return result


def launch(c,provenance):
    if c.get('stage2_training_dispatch_allowed') is not True:raise ValueError('explicit Stage2 dispatch required')
    fn=FunctionType(cached.launch.__code__,dict(cached.launch.__globals__,validate=validate,command=command))
    return fn(dict(c,finite_training_dispatch_allowed=True),provenance)


def run(c,provenance):
    root=Path(c['operations_root'])/'finalization';root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'config.json',c);commit_json(root/'provenance.json',provenance)
    try:
        collected=launch(c,provenance)
        if collected.get('status')!='collected_independent_cached_training_audit_required':
            raise ValueError('complete Stage2 collection required')
        from tools.reports.context_cached_acceptance_sequence import audit_config
        bound=audit_config(c,c['reference_audit']);commit_json(root/'audit-config.json',bound)
        proof=audit(bound,provenance)
        kind=c['training_package']['kind'];updates={'tail':2,'efficiency':100,'formal':634}[kind]
        if proof.get('status')!='stage2_objective_'+kind+'_audited' or len(proof.get('updates',[]))!=updates:
            raise ValueError('complete independent Stage2 update audit required')
        if kind=='formal' and proof.get('real_query_exposures')!=81026:
            raise ValueError('full source exposure budget required')
        commit_json(root/'audit.json',proof)
        commit_json(root/'terminal.json',dict(exit_code=0,status=proof['status']));return proof
    except BaseException as e:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(e),restart=False,preserve_handles=True));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['validate','run','audit'],default='validate');a=p.parse_args();c,prov=load_config(a.config,a.local)
    if a.action=='run':result=run(c,prov)
    elif a.action=='audit':result=audit(c,prov)
    else:validate(c);result=dict(status='stage2_objective_validated_not_executed')
    print(json.dumps(result,indent=2))
