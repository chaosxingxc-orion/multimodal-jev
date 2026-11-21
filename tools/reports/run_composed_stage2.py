"""Frozen-source Stage2 prefix/recovery/tail -> formal; no efficiency rerun."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.source_snapshot import verify
from tools.reports.context_content_audit_engine import ContentAuditEngine
from tools.reports.context_async_pipeline import ReadyQueue
from tools.reports.context_stage2_release_components import graph
from tools.reports.prepare_objective_training_candidate import read_bound
from tools.reports.context_release_deployment import inherited
from tools.reports.context_cached_formal_successor import monitored_training
from tools.reports.run_stage2_objective_pipeline import wait_curve,cpu_evidence


def read(path):return json.loads(Path(path).read_text())


def validate_request(c):
    # This published trainer and its auditor still require full-source exposures.
    # Refuse a data-only view anywhere in the composed deployment.
    def has_view(x):
        if isinstance(x,dict):
            return ('training_dataset' in x or x.get('component')=='stage2_difficulty_release_v2'
                    or any(has_view(v) for v in x.values()))
        return isinstance(x,list) and any(has_view(v) for v in x)
    if has_view(c) or c.get('training_dispatch_allowed') is False:
        raise ValueError('difficulty dataset is data-only; no admitted dynamic Stage2 runtime')
    if (c.get('stage2_training_dispatch_allowed') is not True or c['monitor_interval_seconds']!=900
            or set(c['stages'])!={'prefix','recovery','tail','formal'}
            or c['efficiency']['mode']!='sample_formal_only'):
        raise ValueError('explicit four-stage dispatch and formal-only efficiency required')
    snapshot=c['source_snapshot']
    if Path(__file__).resolve().parents[2]!=Path(snapshot['root']).resolve():
        raise ValueError('controller must execute from frozen source')
    paths=[Path(c['output'])];remote=[];tags=[]
    for stage in c['stages'].values():
        if set(stage)!={'package','remote_root','operations_root','tag'}:raise ValueError('explicit stage locations required')
        paths.extend(Path(stage[k]) for k in ('package','operations_root'))
        remote.append(Path(stage['remote_root']));tags.append(stage['tag'])
    for items in (paths,remote):
        for i,p in enumerate(items):
            if not p.is_absolute() or any(p.is_relative_to(q) or q.is_relative_to(p) for q in items[:i]):
                raise ValueError('disjoint stage locations required')
    if len(set(tags))!=4:raise ValueError('unique run tags required')
    check_paths({str(i):p for i,p in enumerate(paths)})


def validate(c):
    validate_request(c)
    validate_evidence(c)


def validate_evidence(c):
    snapshot=c['source_snapshot']
    verify(snapshot['root'],manifest_sha256=snapshot['manifest_sha256'],io_workers=c['io_workers'])
    for pin in c['pins']:read_bound(pin)
    if c.get('completed_stages'):
        prior=c['completed_auditor_source']
        verify(prior['root'],manifest_sha256=prior['manifest_sha256'],io_workers=c['io_workers'])
    cpu_evidence(c)
    if c.get('completed_prefix'):
        bound=json.loads(read_bound(c['completed_prefix']))
        for key in ('package','operations_root','tag'):
            if bound['deployment'][key]!=c['prefix_operations'][key]:
                raise ValueError('completed prefix must retain exact original deployment')


def run(c,provenance):
    validate_request(c);root=Path(c['output']);root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'config.json',c);commit_json(root/'provenance.json',provenance)
    queue=ReadyQueue(root/'events')
    try:
        validate_evidence(c)
        with ContentAuditEngine(cpu_workers=c['cpu_workers'],io_workers=c['io_workers']) as engine:
            from tools.reports.context_backend_trace_replay import TraceReplayEngine
            replay=TraceReplayEngine(c['trace_replay']) if c.get('trace_replay') else None
            from tools.reports.context_completed_stage_receipts import CompletedStageReceipts
            completed=CompletedStageReceipts(c['completed_stages'],workers=c['io_workers']) if c.get('completed_stages') else None
            views=graph(engine,queue,replay.audit if replay else None,completed=completed)
            def view(name):return views['tools.reports.'+name]
            def execute(kind,spec,ops,module,preparer):
                commit_json(root/(kind+'-preparation.json'),spec)
                spec['manifest_sha256']=preparer.prepare(spec,provenance)['manifest_sha256']
                stage=c['stages'][kind]
                ops.update(training_package=spec,package=stage['package'],manifest_sha256=spec['manifest_sha256'],
                           operations_root=stage['operations_root'],tag=stage['tag'])
                commit_json(root/(kind+'-deployment.json'),ops)
                monitor=root/kind;monitor.mkdir()
                result=monitored_training(ops,provenance,monitor,run_stage=module.run)
                commit_json(root/(kind+'-result.json'),result)
                return read(Path(ops['operations_root'])/'finalization/audit-config.json')
            stage=c['stages']['prefix'];spec=dict(c['prefix_spec'],output=stage['package'],remote_root=stage['remote_root'])
            spec.pop('manifest_sha256',None);spec.pop('audit_output',None)
            ops=dict(c['prefix_operations'],finite_training_dispatch_allowed=True)
            if completed is not None:
                evidence=dict(completed.bounds)
                ops=dict(evidence['prefix']['deployment'])
                for name,proof in completed.reports.items():commit_json(root/(name+'-result.json'),proof)
            elif c.get('completed_prefix'):
                prior=json.loads(read_bound(c['completed_prefix']))
                proof=view('audit_objective_training').audit(prior,provenance)
                commit_json(root/'prefix-result.json',proof)
                evidence={'prefix':prior}
                ops=dict(prior['deployment'])
            else:
                evidence={'prefix':execute('prefix',spec,ops,view('context_objective_training_operations'),view('prepare_objective_training_candidate'))}
            stage=c['stages']['recovery']
            spec=dict(parent_audit=evidence['prefix'],output=stage['package'],remote_root=stage['remote_root'])
            if completed is None:
                evidence['recovery']=execute('recovery',spec,dict(ops),view('context_objective_recovery_operations'),view('prepare_objective_recovery'))
            for kind in (('formal',) if completed is not None else ('tail','formal')):
                prefix=evidence['prefix'];stage=c['stages'][kind]
                if kind=='formal':
                    policy=read(Path(prefix['training_package']['output'])/'config.json')['source_policy_id']
                    if completed is not None:
                        curve=json.loads(read_bound(c['completed_source_curve']))
                        source=json.loads(read_bound(curve['source_evaluation']))
                        for pin in [source['curve'],source['source_audits'],*[point['result'] for point in source['points']]]:read_bound(pin)
                        if curve['points']!=11 or curve['roles']['stage2_initial_and_reference']['policy_id']!=policy:
                            raise ValueError('complete Stage1 curve with same endpoint required')
                        commit_json(root/'source-curve-admission.json',curve)
                    else:commit_json(root/'source-curve-admission.json',wait_curve(c,policy))
                spec=dict(prefix_package=prefix['training_package'],kind=kind,output=stage['package'],remote_root=stage['remote_root'])
                deployment=dict(inherited(ops),reference_audit=ops['reference_audit'],acceptance=dict(evidence),
                                stage2_training_dispatch_allowed=True)
                evidence[kind]=execute(kind,spec,deployment,view('stage2_objective_run_operations'),view('prepare_stage2_objective_run'))
            commit_json(root/'audit-events.json',completed.events if completed else engine.statistics())
            from tools.reports.context_formal_efficiency import summarize
            run_root=Path(evidence['formal']['collection'])/'run'
            for i,window in enumerate(c['efficiency']['windows']):
                commit_json(root/f'efficiency-sample-{i:02d}.json',summarize(run_root,**window))
        commit_json(root/'terminal.json',dict(exit_code=0,status='stage2_formal_collected_and_audited',
                    updates=len(read(root/'formal-result.json')['updates']),selection_evaluation_pending=True,transfer_evaluation_pending=True))
    except BaseException as error:
        queue.abort();commit_json(root/'terminal.json',dict(exit_code=1,error=repr(error),restart=False,preserve_handles=True));raise


def detach(template,local):
    c,p=load_config(template,local);validate_request(c)
    root=Path(c['output']+'-driver');root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-B','-m','tools.reports.run_composed_stage2','--config',str(Path(template).resolve()),
          '--local',str(Path(local).resolve()),'--action','run','--request-sha256',p['resolved_semantic_sha256']]
    with (root/'driver.log').open('x') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=child.pid,argv=argv,started_unix=time.time(),
                process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19])
    commit_json(root/'started.json',record);return record


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--local',required=True);parser.add_argument('--action',choices=['validate','run','detach'],default='validate')
    parser.add_argument('--request-sha256')
    args=parser.parse_args();c,p=load_config(args.config,args.local)
    if args.request_sha256 is not None and args.request_sha256!=p['resolved_semantic_sha256']:
        raise ValueError('detached experiment configuration changed')
    result=detach(args.config,args.local) if args.action=='detach' else run(c,p) if args.action=='run' else validate(c)
    print(json.dumps(result,indent=2))
