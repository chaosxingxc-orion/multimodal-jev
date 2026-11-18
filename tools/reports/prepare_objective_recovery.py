"""Reuse audited plugin step1 state for a new-process step2 replay package.

No NPU dispatch. Preserve the exact plugin identity, original forward source,
Stage1 reference, and trainable-only recovery assets. The parent must pass the
complete objective training auditor before any recovery package is written.
"""
import argparse
import inspect
import json
from types import FunctionType, SimpleNamespace

from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports import prepare_context_cached_recovery as cached
from tools.reports.audit_objective_training import audit as audit_parent


def expected_contract(config,mounted):
    original=cached.expected_contract(config,mounted)
    plugin=config['objective_plugin']
    # Package and parent audit verify the file itself; identity excludes paths
    # so moving an unchanged plugin to the recovery directory is allowed.
    return dict(original,objective_plugin=dict(api='jev-stage2-objective-v1',
        sha256=plugin['sha256'],options=plugin['options']))


def parent_admission(spec,provenance):
    # The output auditor already replays original package/reference provenance
    # and actual outputs, including the objective contract and trace evidence.
    result=audit_parent(spec,provenance)
    if result.get('objective_trace_audit',{}).get('status')!='finite_objective_traces_replayed':
        raise ValueError('complete parent actual-objective trace audit required')
    return result


def replay_runner(manifest):
    source=cached.replay_runner(manifest)
    before="str(r/'source/tools/reports/context_cached_reference_trainer.py')"
    if source.count(before)!=1:raise ValueError('recovery actor entry anchor changed')
    return source.replace(before,"str(r/'objective-extension/bootstrap.py')")


def implementation():
    # Clone the existing deterministic preparation/reconstruction workflow.
    # Functions share this private namespace; live original functions/globals
    # remain untouched, including nested configured/payload/inputs calls.
    namespace=dict(vars(cached))
    for name,value in vars(cached).items():
        if inspect.isfunction(value) and value.__module__==cached.__name__:
            clone=FunctionType(value.__code__,namespace,value.__name__,value.__defaults__,value.__closure__)
            clone.__kwdefaults__=value.__kwdefaults__;namespace[name]=clone
    namespace.update(expected_contract=expected_contract,audit_parent=parent_admission,replay_runner=replay_runner)
    source=inspect.getsource(cached.inputs)
    before="report['status']!='finite_cached_reference_training_outputs_audited'"
    if source.count(before)!=1:raise ValueError('parent objective admission anchor changed')
    source=source.replace(before,"report['status']!='finite_objective_training_outputs_audited'")
    exec(compile(source,'<objective-recovery-inputs>','exec'),namespace)
    return SimpleNamespace(**namespace)


def prepare(config,provenance):
    result=implementation().prepare(config,provenance)
    return dict(result,objective_identity_preserved=True,npu_executed=False)


def audit(config,provenance):
    result=implementation().audit(config,provenance)
    return dict(result,status='objective_recovery_independently_bound_not_npu_admitted',
                objective_identity_preserved=True,npu_executed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--local',required=True);p.add_argument('--action',choices=['prepare','audit'],default='prepare')
    args=p.parse_args();c,provenance=load_config(args.config,args.local)
    result=prepare(c,provenance) if args.action=='prepare' else audit(c,provenance)
    if args.action=='audit':commit_json(c['audit_output'],result)
    print(json.dumps(result,indent=2))
