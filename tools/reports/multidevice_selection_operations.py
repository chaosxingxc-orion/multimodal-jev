"""Configured TP1 evaluation using the existing exact lifecycle and scorers."""
import argparse
import json
from pathlib import Path
from types import FunctionType

from multimodal_jev.local_config import load_config
from tools.reports import context_replica_selection_operations as old
from tools.reports import context_train_parity_operations as lifecycle
from tools.reports import audit_context_train_parity as transport
from tools.reports import prepare_multidevice_selection as preparation
from tools.reports.context_multidevice_evaluation import command,transformed,visibility,audit_macros


def validate(c):
    spec=c['selection_package'];preparation.audit(spec,{'source':'multidevice_lifecycle'})
    checker=FunctionType(lifecycle.validate.__code__,dict(lifecycle.validate.__globals__,
        audit_package=lambda *_:None,command=command))
    result=checker(dict(c,parity_package=spec));visibility(result[0])
    from tools.reports.context_release_deployment import audit as audit_release
    audit_release(command(c,result[0],result[1]),result[1])
    return result


def launch(c,provenance):
    if c.get('selection_dispatch_allowed') is not True:raise ValueError('explicit selection authorization required')
    launch_fn=FunctionType(lifecycle.launch.__code__,dict(lifecycle.launch.__globals__,validate=validate,command=command))
    result=launch_fn(dict(c,finite_inference_dispatch_allowed=True),provenance)
    return dict(result,status='selection_collected_independent_audit_required')


def audit_outputs(package,collection):
    # Preserve all original tensor, source, transport, score and identity checks.
    config=old.read(Path(package)/'parity-macro.json')
    fn=transformed(old.audit_outputs,[
        ("supervision.get('child_exit_codes')!=[0,0,0]", "supervision.get('child_exit_codes')!=[0]*(len(m['inference_services'])+1)"),
        ('macro_queries=128)', "macro_queries=c['macro_queries'])"),
        ('replicas=2)', "replicas=len(pool['replicas']))"),
    ],audit_macros=lambda *a,**kw:audit_macros(*a,replicas=len(config['macro_client']['replicas']),**kw))
    result=fn(package,collection)
    m=old.read(Path(package)/'PREPARED.json');c=old.read(Path(package)/'parity-macro.json')
    visibility(m)
    # Each service must have contributed at least one independently selected
    # HF-panel query. Aggregate parity alone must not mask an untested worker.
    selected=old.read(Path(collection)/'hf/panel.json');panel_ids={a['id'] for a in selected['assets']}
    inference=Path(collection)/'evaluation/inference';pool=c['macro_client']
    journal=Path(collection)/'selection-journal'/pool['attempt_id'];coverage=[0]*len(pool['replicas'])
    for i,b in enumerate(old.read(inference/'result.json')['batches']):
        records=old.read(inference/b['path'])['predictions']
        for job in old.read(journal/f'evaluation-{i:08d}'/'INPUT.json')['jobs']:
            coverage[job['replica_id']]+=sum(records[index]['id'] in panel_ids for index in job['indices'])
    if any(n==0 for n in coverage):raise ValueError('HF panel must cover every inference service')
    return dict(result,evaluation_replicas=len(coverage),per_service_hf_queries=coverage,
        macro_queries=c['macro_queries'],rollout_queries=pool['rollout_queries'])


def audit(c,provenance):
    fn=FunctionType(transport.audit.__code__,dict(transport.audit.__globals__,
        audit_package=preparation.audit,audit_outputs=audit_outputs,command=command))
    result=fn(dict(c,parity_package=c['selection_package']),provenance)
    result['scope']='Full SLURP selection; configured TP1 replicas with HF panel coverage for every service'
    return result


def run(c,provenance):
    return FunctionType(old.run.__code__,dict(old.run.__globals__,launch=launch,audit=audit))(c,provenance)


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['validate','run','audit'],default='validate');a=p.parse_args()
    c,provenance=load_config(a.config,a.local)
    if a.action=='run':result=run(c,provenance)
    elif a.action=='audit':result=audit(c,provenance)
    else:
        m,conf,_=validate(c);result=dict(status='multidevice_evaluation_validated_not_executed',argv=command(c,m,conf),npu_executed=False)
    print(json.dumps(result,indent=2))
