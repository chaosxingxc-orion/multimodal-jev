"""Versioned async controller with shared parallel Stage1 source audits.

NPU runtime/metrics and every admission predicate remain frozen. Only the
Stage1 selection branch uses the validated host-side audit engine. Stage2
reference/GRPO auditors remain original until separately validated.
"""
import argparse
import fcntl
import inspect
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
from types import FunctionType,SimpleNamespace

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports import context_async_pipeline as original
from tools.reports.context_shared_source_audit import SourceAuditEngine


def optimized_selection(engine):
    from tools.reports import context_replica_selection_operations as point
    from tools.reports import collect_context_replica_infonce_curve as curve
    preparation=engine.preparation()
    validate=FunctionType(point.validate.__code__,dict(point.validate.__globals__,preparation=preparation))
    launch=FunctionType(point.launch.__code__,dict(point.launch.__globals__,validate=validate))
    audit=FunctionType(point.audit.__code__,dict(point.audit.__globals__,preparation=preparation))
    run=FunctionType(point.run.__code__,dict(point.run.__globals__,launch=launch,audit=audit))
    operations=SimpleNamespace(validate=validate,launch=launch,audit=audit,run=run)
    reconstruct=FunctionType(curve.reconstruct.__code__,dict(curve.reconstruct.__globals__,
        point=operations,reconstruct_plan=engine.reconstruct))
    collector=SimpleNamespace(collect=lambda c,p:curve.publication.collect(c,p,reconstructor=reconstruct))
    body=inspect.getsource(original.selection_curve)
    replacements={
        '        from tools.reports import prepare_context_replica_selection as preparation':'        preparation=optimized_preparation',
        '        from tools.reports import context_replica_selection_operations as point':'        point=optimized_point',
        '        from tools.reports import collect_context_replica_infonce_curve as curve':'        curve=optimized_curve',
    }
    for before,after in replacements.items():
        if body.count(before)!=1:raise ValueError('selection adapter anchor changed')
        body=body.replace(before,after)
    scope=dict(original.selection_curve.__globals__,optimized_preparation=preparation,
               optimized_point=operations,optimized_curve=collector)
    exec(compile(body,'<shared-audit-selection-pipeline>','exec'),scope)
    return scope['selection_curve'],operations


def validate(c):
    old=original.validate(c)
    prior=original.checked_source(c['previous_controller'])
    record=original.checked_source(c['previous_driver'])
    if (c['previous_driver']['path']!=str(Path(prior['output']).with_name(Path(prior['output']).name+'-driver')/'started.json')
            or c['sources']!=prior['sources'] or c['deployment']!=prior['deployment']
            or c['preprepared_stage1_endpoint']!=prior['preprepared_stage1_endpoint']):
        raise ValueError('exact original async controller and unchanged sources required')
    for key in ('output','prepared_root','operations_root','remote_root'):
        new,previous=Path(c[key]).resolve(),Path(prior[key]).resolve()
        if new.is_relative_to(previous) or previous.is_relative_to(new):raise ValueError('isolated successor roots required')
    gate=original.checked_source(c['audit_engine_admission'])
    if (gate['status']!='shared_source_and_selection_chain_verified' or gate['full_proof_equal'] is not True
            or gate['batch_computations']!=1 or gate['batch_reuses']<3 or gate['model_calls']!=0):
        raise ValueError('full source/selection exact CPU admission required')
    evidence=original.checked_source(c['audit_engine_config'])
    for name,digest in evidence['pins'].items():
        if file_hash(name)!=digest:raise ValueError('admitted engine dependency changed: '+name)
    expected=original.checked_source(evidence['source_plan'])
    for binding in c['audit_engine_results']:
        if original.checked_source(binding)!=expected:
            raise ValueError('actual complete differential audit evidence differs')
    if len(c['audit_engine_results'])!=2:
        raise ValueError('both independently reconstructed full proofs required')
    return old


def retire(c,old):
    """Retire one exact CPU-only controller, preserving all old evidence."""
    prior=original.checked_source(c['previous_controller']);record=original.checked_source(c['previous_driver'])
    fd=os.pidfd_open(record['pid']);stopped=False
    try:
        original.exact_process(record);signal.pidfd_send_signal(fd,signal.SIGSTOP);stopped=True
        deadline=time.monotonic()+2
        while original.exact_process(record) not in ('T','t'):
            if time.monotonic()>deadline:raise ValueError('controller did not stop')
            time.sleep(.01)
        proc=Path('/proc')/str(record['pid'])
        for task in proc.joinpath('task').iterdir():
            if (task/'children').read_text().strip():raise ValueError('controller has active child work; retain queue')
        if list(Path(prior['operations_root']).glob('**/*-attempt.json')):
            raise ValueError('previous NPU attempts exist; explicit reconciliation required')
        if (Path(prior['output'])/'terminal.json').exists():raise ValueError('previous controller already terminal')
        dep=c['deployment']
        with Path(dep['lease']).open('r') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            check=subprocess.run(['ssh','-o','ConnectTimeout=10',dep['host'],'npu-smi info'],
                capture_output=True,text=True,check=True,timeout=55)
            if any(f'No running processes found in NPU {n}' not in check.stdout for n in (4,5,6,7)):
                raise ValueError('four-card resource occupied; retain queue')
            root=Path(c['output'])
            commit_json(root/'retirement-intent.json',dict(previous=record,reason='validated_parallel_audit_successor',unix=time.time()))
            signal.pidfd_send_signal(fd,signal.SIGTERM);signal.pidfd_send_signal(fd,signal.SIGCONT);stopped=False
            poll=select.poll();poll.register(fd,select.POLLIN)
            if not poll.poll(10000):raise ValueError('previous controller failed to exit')
            commit_json(root/'retirement-complete.json',dict(previous=record,old_artifacts_preserved=True,
                remote_processes_signalled=False,unix=time.time()))
    finally:
        if stopped:
            try:signal.pidfd_send_signal(fd,signal.SIGCONT)
            except ProcessLookupError:pass
        os.close(fd)


def run(c,p):
    engine=SourceAuditEngine(cpu_workers=c['cpu_workers'],io_workers=c['io_workers'])
    selection,_=optimized_selection(engine)
    fn=FunctionType(original.run.__code__,dict(original.run.__globals__,validate=validate,retire=retire,selection_curve=selection))
    try:return fn(c,p)
    finally:
        root=Path(c['output'])
        if root.exists():
            commit_json(root/'audit-engine-statistics.json',dict(batch_computations=engine.store.executions,
                batch_reuses=engine.store.hits,events=engine.events,model_calls=0))


def detach(template,local):
    c,_=load_config(template,local);validate(c)
    root=Path(c['output']).with_name(Path(c['output']).name+'-driver')
    check_paths({'driver':root});root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-B','-m','tools.reports.context_audit_optimized_pipeline',
        '--config',str(Path(template).resolve()),'--local',str(Path(local).resolve()),'--action','run']
    with (root/'driver.log').open('x') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=child.pid,argv=argv,started_unix=time.time(),
        process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19])
    commit_json(root/'started.json',record);return record


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['validate','detach','run'],default='validate');a=p.parse_args()
    c,provenance=load_config(a.config,a.local)
    if a.action=='detach':result=detach(a.config,a.local)
    elif a.action=='run':result=run(c,provenance)
    else:validate(c);result=dict(status='validated_only',npu_executed=False)
    print(json.dumps(result,indent=2))
