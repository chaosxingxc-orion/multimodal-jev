"""Finite plugin training audit layered on the unchanged cached-reference auditor.

Keep source, truth, rollout, reference, export, optimizer and transport checks.
Add checkpoint objective identity and complete actual objective trace replay.
This is not a dispatcher or a fresh-process recovery admission.
"""
import argparse
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.objective_plugin import ObjectivePlugin
from multimodal_jev.context_topology_batch_plan import derive
from tools.reports import audit_context_cached_training as cached
from tools.reports.audit_objective_traces import audit as audit_traces
from tools.reports.prepare_objective_training_candidate import audit as audit_package
from tools.reports.context_objective_storage import command


def read(path):
    return json.loads(Path(path).read_text())


def packaged_plugin(package, config):
    remote=Path(read(Path(package)/'PREPARED.json')['remote_root'])
    spec=config['objective_plugin']
    relative=Path(spec['path']).relative_to(remote)
    if relative.as_posix()!='objective-extension/extensions/stage2_objective.py':
        raise ValueError('packaged objective extension required')
    return dict(spec,path=str(Path(package)/relative))


def audit_outputs(package,collection,*,reference_root,loss_atol):
    package,collection=Path(package),Path(collection)
    config=read(package/'config.json')
    if (config.get('objective_trace') is not True
            or config['experiment_kind']!='engineering_prefix'
            or config['start_update']!=0 or config['stop_update']!=2):
        raise ValueError('complete traced two-update prefix required')
    spec=packaged_plugin(package,config);plugin=ObjectivePlugin(spec)

    def expected_contract(c,mounted):
        result=cached.expected_contract(c,mounted)
        return dict(result,objective_plugin=plugin.identity)

    # Scoped module view: never mutate the shared auditor used by live jobs.
    shared=SimpleNamespace(**vars(cached.shared))
    shared.audit_outputs=FunctionType(cached.shared.audit_outputs.__code__,
        dict(cached.shared.audit_outputs.__globals__,expected_contract=expected_contract))
    base=FunctionType(cached.audit_outputs.__code__,
        dict(cached.audit_outputs.__globals__,shared=shared,expected_contract=expected_contract))
    original=base(package,collection,reference_root=reference_root)
    plan=derive(config['batch_config']);world=plan['dp_replicas']
    if original['world_size']!=world or len(original['updates'])!=2:
        raise ValueError('complete original training audit required before trace replay')
    run=collection/'run';updates=[];expected_paths=set()
    for update in range(2):
        groups=run/f'groups-{update:04d}.pt';traces=[]
        for rank in range(world):
            path=run/f'rank-{rank}/objective-traces/objective-{update:04d}.json'
            expected_paths.add(path)
            traces.append(dict(rank=rank,path=str(path),sha256=file_hash(path)))
        updates.append(dict(update=update,groups=dict(path=str(groups),sha256=file_hash(groups)),traces=traces))
    actual_paths=set(run.glob('rank-*/objective-traces/**/*'))
    if actual_paths!=expected_paths:
        raise ValueError('exact complete objective trace file set required')
    replay=audit_traces(dict(objective_plugin=spec,world_size=world,
        local_batch=plan['local_effective_batch'],microbatch=config['batch_config']['microbatch'],
        updates=updates,loss_atol=loss_atol,output='unused'),{'source':'independent_training_audit'})
    return dict(original,status='finite_objective_training_outputs_audited',
        objective_trace_audit=replay,objective_plugin=plugin.identity,
        fresh_process_replay_verified=False,training_dispatch_allowed=False)


def audit(c,provenance):
    specification=c['training_package']
    # The legacy wrapper only needs this producer pointer for reference lookup.
    # Package reconstruction always consumes the actual objective specification.
    legacy=dict(c,training_package=dict(specification,
        reference_job=specification['parent']['reference_job']))
    def package_audit(_spec,prov):
        return audit_package(specification,prov)
    def outputs(package,collection,*,reference_root):
        return audit_outputs(package,collection,reference_root=reference_root,
                             loss_atol=c['objective_loss_atol'])
    checker=FunctionType(cached.audit.__code__,dict(cached.audit.__globals__,
        audit_package=package_audit,audit_outputs=outputs,command=command))
    return dict(checker(legacy,provenance),formal_training_admitted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    args=p.parse_args();c,provenance=load_config(args.config,args.local)
    result=audit(c,provenance);commit_json(c['output'],result)
    print(json.dumps(result,indent=2))
