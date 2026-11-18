"""Fresh cached GRPO replay through the existing lease/wait/collect lifecycle."""
import argparse
import json
from pathlib import Path
from types import FunctionType

from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports import context_cached_training_operations as shared
from tools.reports.prepare_context_cached_recovery import audit as audit_package
from tools.reports.audit_context_cached_recovery import audit


def validate(c):
    audit_package(c['training_package'],{'source':'cached_recovery_lifecycle'})
    validator=FunctionType(shared.validate.__code__,dict(shared.validate.__globals__,admission=lambda c:None),
                           'validate_cached_recovery')
    return validator(c)


def launch(c,provenance):
    # Original cached lifecycle still enforces explicit finite-training flag,
    # same four-card lease, readonly inputs, bounded preflight and exact wait.
    launcher=FunctionType(shared.launch.__code__,dict(shared.launch.__globals__,validate=validate),
                          'launch_cached_recovery')
    return launcher(c,provenance)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['validate','launch','audit'],default='validate');a=p.parse_args()
    c,provenance=load_config(a.config,a.local)
    if a.action=='launch':result=launch(c,provenance)
    elif a.action=='audit':result=audit(c,provenance);commit_json(c['output'],result)
    else:
        m,config,_=validate(c)
        result=dict(status='cached_recovery_validated_not_executed',argv=shared.command(c,m,config),
                    training_dispatch_allowed=False)
    print(json.dumps(result,indent=2))
