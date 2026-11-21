"""Audit plugin recovery outputs and compare step2 with continuous training."""
from pathlib import Path
from types import FunctionType, SimpleNamespace

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_topology_batch_plan import derive
from tools.jev.context_replay_audit import require_exact
from tools.reports import audit_context_cached_recovery as cached
from tools.reports import prepare_objective_recovery as preparation
from tools.reports.audit_objective_training import packaged_plugin
from tools.reports.audit_objective_traces import audit as audit_traces
from tools.reports.context_objective_storage import command


def audit_outputs(package,collection,*,reference_root,loss_atol):
    package,collection=Path(package),Path(collection)
    config=cached.cached.read(package/'config.json')
    if config.get('objective_trace') is not True:
        raise ValueError('complete recovery objective traces required')
    view=SimpleNamespace(**vars(cached.cached))
    view.expected_contract=preparation.expected_contract
    original=FunctionType(cached.audit_outputs.__code__,
        dict(cached.audit_outputs.__globals__,cached=view))
    result=original(package,collection,reference_root=reference_root)
    plan=derive(config['batch_config']);world=plan['dp_replicas'];run=collection/'run'
    groups=run/'groups-0001.pt';traces=[];expected=set()
    for rank in range(world):
        p=run/f'rank-{rank}/objective-traces/objective-0001.json';expected.add(p)
        traces.append(dict(rank=rank,path=str(p),sha256=file_hash(p)))
    if set(run.glob('rank-*/objective-traces/**/*'))!=expected:
        raise ValueError('exact recovery objective trace population required')
    proof=audit_traces(dict(objective_plugin=packaged_plugin(package,config),world_size=world,
        local_batch=plan['local_effective_batch'],microbatch=config['batch_config']['microbatch'],
        updates=[dict(update=1,groups=dict(path=str(groups),sha256=file_hash(groups)),traces=traces)],
        loss_atol=loss_atol,output='unused'),{'source':'objective_recovery_output_audit'})
    return dict(result,status='objective_recovery_outputs_audited',objective_trace_audit=proof)


def compare(continuous,replayed,*,world_size=2):
    result=cached.compare(continuous,replayed,world_size=world_size)
    for rank in range(world_size):
        name=f'rank-{rank}/objective-traces/objective-0001.json'
        require_exact(cached.cached.read(Path(continuous)/name),cached.cached.read(Path(replayed)/name),
                      'actual_objective/'+name)
    return dict(result,exact_objective_traces=True)


def audit(c,provenance):
    spec=c['training_package'];parent=spec['parent_audit']
    _,_,_,run,parent_report,_,_=preparation.implementation().inputs(spec,provenance)
    bound=dict(c,training_package=dict(spec,
        reference_job=parent['training_package']['parent']['reference_job']))
    def package_check(_spec,prov):return preparation.audit(spec,prov)
    def outputs(package,collection,*,reference_root):
        return audit_outputs(package,collection,reference_root=reference_root,loss_atol=c['objective_loss_atol'])
    checker=FunctionType(cached.cached.audit.__code__,dict(cached.cached.audit.__globals__,
        audit_package=package_check,audit_outputs=outputs,command=command))
    result=checker(bound,provenance)
    if result['container_id']==parent_report['container_id']:
        raise ValueError('fresh container required for objective recovery acceptance')
    return dict(result,replay=compare(run,Path(c['collection'])/'run'),
        status='finite_objective_recovery_audited',fresh_process_npu_resume_verified=True,
        formal_training_admitted=False,training_dispatch_allowed=False)
