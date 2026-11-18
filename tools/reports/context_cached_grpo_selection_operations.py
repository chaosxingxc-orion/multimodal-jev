"""GRPO policy origin over the shared complete-selection lifecycle and scorer."""
import argparse
import json
from pathlib import Path
from types import FunctionType

from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports import context_replica_selection_operations as shared
from tools.reports import prepare_context_cached_grpo_selection as preparation


def validate(c):
    return FunctionType(shared.validate.__code__,dict(shared.validate.__globals__,preparation=preparation))(c)


def launch(c,provenance):
    return FunctionType(shared.launch.__code__,dict(shared.launch.__globals__,validate=validate))(c,provenance)


def audit_outputs(package,collection):
    result=shared.audit_outputs(package,collection)
    return dict(result,status='complete_replica_cached_grpo_selection_audited')


def audit(c,provenance):
    fn=FunctionType(shared.audit.__code__,dict(shared.audit.__globals__,preparation=preparation,audit_outputs=audit_outputs))
    return fn(c,provenance)


def run(c,provenance):
    if c.get('selection_dispatch_allowed') is not True:raise ValueError('explicit full selection authorization required')
    root=Path(c['operations_root'])/'finalization';check_paths({'selection_finalization':root})
    root.mkdir(parents=True,exist_ok=False);commit_json(root/'config.json',c);commit_json(root/'provenance.json',provenance)
    try:
        result=launch(c,provenance)
        if result['status']!='selection_collected_independent_audit_required':raise ValueError('full selection collection required')
        stem=Path(c['operations_root'])/c['tag']
        bound=dict(selection_package=c['selection_package'],deployment=c,
            **{k:str(stem)+suffix for k,suffix in (
                ('launch','-launch.json'),('inspect','-inspect.json'),('wait','-wait.json'),
                ('transport','-collected.transport.json'),('collection','-collected'),('archive','-collected.tar'))})
        commit_json(root/'audit-config.json',bound);proof=audit(bound,provenance)
        if proof['status']!='complete_replica_cached_grpo_selection_audited':raise ValueError('independent complete GRPO selection audit required')
        commit_json(root/'audit.json',proof)
        commit_json(root/'terminal.json',dict(exit_code=0,status='grpo_selection_point_collected_and_audited',source_step=proof['source_step']))
        return proof
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False,preserve_handles=True));raise


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['validate','run','audit'],default='validate');a=p.parse_args();c,provenance=load_config(a.config,a.local)
    if a.action=='run':result=run(c,provenance)
    elif a.action=='audit':result=audit(c,provenance)
    else:validate(c);result=dict(status='grpo_selection_lifecycle_validated_not_executed',npu_executed=False)
    print(json.dumps(result,indent=2))
