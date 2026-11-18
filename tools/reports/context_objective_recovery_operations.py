"""Run one fresh-process plugin replay with the same lease and collection."""
import argparse
import json
from types import FunctionType

from multimodal_jev.local_config import load_config
from tools.reports import context_objective_training_operations as shared
from tools.reports import prepare_objective_recovery as preparation
from tools.reports.audit_objective_recovery import audit


def admission(c):
    proof=preparation.audit(c['training_package'],{'source':'objective_recovery_lifecycle'})
    shared.bootstrap_evidence(c)
    return proof


def validate(c):
    checker=FunctionType(shared.validate.__code__,dict(shared.validate.__globals__,admission=admission))
    return checker(c)


def launch(c,provenance):
    runner=FunctionType(shared.launch.__code__,dict(shared.launch.__globals__,validate=validate))
    return runner(c,provenance)


def run(c,provenance):
    # Shared finalization calls our recovery auditor before writing success.
    # Its successful prefix label is deliberately replaced in this derived
    # function so a one-step replay cannot be mistaken for a two-step prefix.
    import inspect
    source=inspect.getsource(shared.run)
    old="status='objective_prefix_collected_and_audited'"
    if source.count(old)!=1:raise ValueError('objective finalization anchor changed')
    source=source.replace(old,"status='objective_recovery_collected_and_audited'")
    if source.count('recovery_admitted=False')!=1:
        raise ValueError('objective recovery terminal anchor changed')
    source=source.replace('recovery_admitted=False','recovery_admitted=True')
    scope=dict(shared.run.__globals__,launch=launch,audit_training=audit)
    exec(compile(source,'<objective-recovery-finalization>','exec'),scope)
    return scope['run'](c,provenance)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--local',required=True);p.add_argument('--action',choices=['validate','run'],default='validate')
    args=p.parse_args();c,provenance=load_config(args.config,args.local)
    if args.action=='run':result=run(c,provenance)
    else:
        m,config,_=validate(c)
        result=dict(status='objective_recovery_validated_not_executed',argv=shared.command(c,m,config),
                    npu_executed=False,training_dispatch_allowed=False)
    print(json.dumps(result,indent=2))
