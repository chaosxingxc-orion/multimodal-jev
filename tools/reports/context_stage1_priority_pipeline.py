"""Validated replacement: Stage1 curve first, isolated source audit process.

Retains full downstream efficiency/GRPO/transfer plan and original gates.
Replaces only a CPU controller with no NPU attempt; refuses active NPU work.
"""
import argparse
import fcntl
import inspect
import json
import os
import re
from pathlib import Path
import select
import signal
import shlex
import subprocess
import sys
import time
from types import FunctionType

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports import context_async_pipeline as original
from tools.reports.context_audit_optimized_pipeline import optimized_selection
from tools.reports.context_isolated_source_audit import IsolatedSourceAuditEngine


def validate(c):
    old = original.validate(c)
    prior = original.checked_source(c['previous_controller'])
    if any(c[k] != prior[k] for k in ('sources', 'deployment', 'preprepared_stage1_endpoint')):
        raise ValueError('same experimental dependencies required')
    for key in ('output','prepared_root','operations_root','remote_root'):
        a,b=Path(c[key]).resolve(),Path(prior[key]).resolve()
        if a.is_relative_to(b) or b.is_relative_to(a):
            raise ValueError('isolated successor paths required')
    record=original.checked_source(c['previous_driver'])
    if (c['previous_driver']['path'] != str(Path(prior['output']).with_name(Path(prior['output']).name+'-driver')/'started.json')
            or 'tools.reports.context_audit_optimized_pipeline' not in record['argv']):
        raise ValueError('exact prior controller required')
    evidence=original.checked_source(c['isolation_config'])
    report=original.checked_source(c['isolation_admission'])
    if (report['status']!='isolated_source_selection_audited' or report['full_source_equal'] is not True
            or report['full_selection_equal'] is not True or report['worker_closed'] is not True):
        raise ValueError('full isolated source/selection proof required')
    for name,digest in evidence['pins'].items():
        if file_hash(name)!=digest:raise ValueError('admitted isolation code changed: '+name)
    expected=original.checked_source(evidence['source_plan'])
    for name in c['source_results']:
        if original.checked_source(name)!=expected:raise ValueError('full isolated source proof differs')
    if len(c['source_results'])!=2:raise ValueError('both full proofs required')
    if original.checked_source(c['selection_result'])!=original.checked_source(evidence['selection_expected']):
        raise ValueError('actual isolated selection proof differs')
    if c.get('adopt_selection'):
        adopted=c['adopt_selection'];ops=original.checked_source(adopted['operations'])
        launch=original.checked_source(adopted['launch'])
        attempt=original.checked_source(adopted['attempt'])
        if (ops['selection_package']['source_step']!=318 or ops['host']!=c['deployment']['host']
                or ops['lease']!=c['deployment']['lease'] or attempt['config']!=dict(ops,finite_training_dispatch_allowed=True)
                or launch['manifest_sha256']!=ops['manifest_sha256']
                or launch['remote_root']!=ops['selection_package']['remote_root']
                or adopted['launch']['path']!=str(Path(ops['operations_root'])/(ops['tag']+'-launch.json'))):
            raise ValueError('exact existing endpoint318 attempt required')
    return old


def tracker_only(pid, parent):
    proc=Path('/proc')/str(pid)
    fields=(proc/'stat').read_text().split(') ',1)[1].split()
    argv=(proc/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
    return (int(fields[1])==parent and len(argv)==4 and argv[:3]==[sys.executable,'-B','-c']
        and re.fullmatch(r'from multiprocessing\.resource_tracker import main;main\(\d+\)',argv[3]) is not None)


def retire(c, old):
    prior=original.checked_source(c['previous_controller']);record=original.checked_source(c['previous_driver'])
    fd=os.pidfd_open(record['pid']);stopped=False
    try:
        original.exact_process(record)
        signal.pidfd_send_signal(fd,signal.SIGSTOP);stopped=True
        deadline=time.monotonic()+2
        while original.exact_process(record) not in ('T','t'):
            if time.monotonic()>deadline:raise ValueError('controller did not stop')
            time.sleep(.01)
        children={int(v) for task in (Path('/proc')/str(record['pid'])/'task').iterdir()
                  for v in (task/'children').read_text().split()}
        adopted=c.get('adopt_selection');waiters=[]
        handle=original.checked_source(adopted['launch']) if adopted else None
        def owned_waiter(pid):
            proc=Path('/proc')/str(pid)
            argv=(proc/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
            expected=['ssh','-o','ConnectTimeout=10',c['deployment']['host'],
                      'docker wait '+handle['container_id']] if handle else []
            if argv==expected:
                waiters.append(pid);return True
            return False
        if any(not tracker_only(pid,record['pid']) and not owned_waiter(pid) for pid in children):
            raise ValueError('active child work requires reconciliation')
        attempts=list(Path(prior['operations_root']).glob('**/*-attempt.json'))
        if {str(p) for p in attempts} != ({adopted['attempt']['path']} if adopted else set()):
            raise ValueError('unreconciled NPU attempt exists')
        if adopted:
            ops=original.checked_source(adopted['operations'])
            stem=Path(ops['operations_root'])/ops['tag']
            if any(Path(str(stem)+suffix).exists() for suffix in ('-collected','-collected.tar','-wait.json')):
                raise ValueError('collection/terminal boundary changed; reconcile again')
            observed=json.loads(subprocess.check_output(['ssh','-o','ConnectTimeout=10',dep_host(c),
                'docker inspect '+handle['container_id']],text=True,timeout=55))[0]
            if (observed['Id']!=handle['container_id'] or observed['Image']!=handle['image']
                    or observed['State']['Status'] not in ('running','exited')):
                raise ValueError('exact running or terminal adopted container required')
        if (Path(prior['output'])/'terminal.json').exists():raise ValueError('prior already terminal')
        dep=c['deployment']
        with Path(dep['lease']).open('r') as lease:
            if not adopted:
                fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                observed=subprocess.run(['ssh','-o','ConnectTimeout=10',dep['host'],'npu-smi info'],
                    capture_output=True,text=True,check=True,timeout=55)
                if any(f'No running processes found in NPU {n}' not in observed.stdout for n in (4,5,6,7)):
                    raise ValueError('NPU occupied; preserve existing task')
            root=Path(c['output'])
            commit_json(root/'retirement-intent.json',dict(previous=record,children=sorted(children),
                adopted_container=handle['container_id'] if handle else None,
                reason='user_authorized_validated_stage1_priority_replacement',unix=time.time()))
            signal.pidfd_send_signal(fd,signal.SIGTERM)
            signal.pidfd_send_signal(fd,signal.SIGCONT);stopped=False
            poll=select.poll();poll.register(fd,select.POLLIN)
            if not poll.poll(10000):raise ValueError('old controller did not exit')
            # Only disconnect exact local read-only docker-wait observers.
            # The remote container is deliberately left running.
            for pid in waiters:
                try:
                    childfd=os.pidfd_open(pid)
                    try:
                        if not owned_waiter_check(pid,dep['host'],handle['container_id']):
                            raise ValueError('waiter identity changed')
                        signal.pidfd_send_signal(childfd,signal.SIGTERM)
                    finally:os.close(childfd)
                except ProcessLookupError:pass
            if adopted:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            commit_json(root/'retirement-complete.json',dict(previous=record,remote_processes_signalled=False,
                old_artifacts_preserved=True,unix=time.time()))
    finally:
        if stopped:
            try:signal.pidfd_send_signal(fd,signal.SIGCONT)
            except ProcessLookupError:pass
        os.close(fd)


def dep_host(c):
    return c['deployment']['host']


def owned_waiter_check(pid,host,cid):
    argv=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
    return argv==['ssh','-o','ConnectTimeout=10',host,'docker wait '+cid]


def adopt_point(c,provenance,queue,point):
    """Wait/collect the original exact container; never create/start a model."""
    from tools.reports.context_replica_training_operations import wait_exact
    from tools.jev.collect_context_replica_training import collect
    from tools.reports.context_dual_probe_operations import check_runtime
    adoption=c['adopt_selection'];ops=original.checked_source(adoption['operations'])
    handle=original.checked_source(adoption['launch']);root=Path(c['operations_root'])/'adopted-infonce-0318'
    final=root/'finalization';final.mkdir(parents=True,exist_ok=False)
    commit_json(final/'config.json',ops)
    def remote(args,bounded=True):
        return subprocess.run(['ssh','-o','ConnectTimeout=10',ops['host'],shlex.join(args)],
            capture_output=True,text=True,timeout=55 if bounded else None)
    with queue.slot(0,'adopted-endpoint318'), Path(ops['lease']).open('r') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        code=wait_exact(remote,handle['container_id'])
        observed=remote(['docker','inspect',handle['container_id']]);observed.check_returncode()
        observed=json.loads(observed.stdout)[0]
        commit_json(root/'wait.json',dict(container_id=handle['container_id'],exit_code=code))
        commit_json(root/'inspect.json',observed)
        if code:queue.abort()
    collect(handle,root/'collected',host=ops['host'])
    check_runtime(dict(argv=handle['argv'],image=handle['image']),handle,observed)
    if code:raise ValueError('adopted endpoint failed; original evidence preserved')
    bound=dict(selection_package=ops['selection_package'],deployment=ops,
        launch=adoption['launch']['path'],inspect=str(root/'inspect.json'),wait=str(root/'wait.json'),
        transport=str(root/'collected.transport.json'),collection=str(root/'collected'),archive=str(root/'collected.tar'))
    commit_json(final/'audit-config.json',bound)
    proof=point.audit(bound,provenance)
    if proof['status']!='complete_replica_infonce_selection_audited' or proof['source_step']!=318:
        raise ValueError('full adopted endpoint audit required')
    commit_json(final/'audit.json',proof)
    commit_json(final/'terminal.json',dict(exit_code=0,status='selection_point_collected_and_audited',source_step=318))
    return dict(step=318,audit_config=str(final/'audit-config.json'),audit_config_sha256=file_hash(final/'audit-config.json'))


def adopted_selection(engine):
    from tools.reports import context_audit_optimized_pipeline as optimized
    # Use the original source so the same substitution remains reviewable.
    selection,point=optimized.optimized_selection(engine)
    body=inspect.getsource(original.selection_curve)
    for before,after in {
        '        from tools.reports import prepare_context_replica_selection as preparation':'        preparation=optimized_preparation',
        '        from tools.reports import context_replica_selection_operations as point':'        point=optimized_point',
        '        from tools.reports import collect_context_replica_infonce_curve as curve':'        curve=optimized_curve',
    }.items():
        if body.count(before)!=1:raise ValueError('selection anchor changed')
        body=body.replace(before,after)
    anchor='    def one(step):\n'
    inserted=(anchor+"        if not grpo and step==318 and c.get('adopt_selection'):\n"
              "            entry=adopt_point(c,provenance,queue,point)\n"
              "            commit_json(root/f'completed-{step:04d}.json',entry)\n"
              "            return entry\n")
    if body.count(anchor)!=1:raise ValueError('selection task anchor changed')
    scope=dict(selection.__globals__,adopt_point=adopt_point)
    exec(compile(body.replace(anchor,inserted),'<adopted-stage1-selection>','exec'),scope)
    return scope['selection_curve']


def adapted_run(selection):
    body=inspect.getsource(original.run)
    begin='        with ThreadPoolExecutor(max_workers=2,thread_name_prefix=\'independent-branches\') as branches:\n'
    end='        from tools.reports import context_cached_grpo_source_plan as source\n'
    if body.count(begin)!=1 or body.count(end)!=1:raise ValueError('priority pipeline anchor changed')
    start,stop=body.index(begin),body.index(end)
    replacement=('        source_cc,source_result=selection_curve(c,template,STEPS,provenance,queue)\n'
                 '        efficiency_result=finite_and_efficiency(c,old,provenance,queue)\n'
                 '        finalization,ops,_=formal_training(c,efficiency_result,source_cc,provenance,queue)\n')
    scope=dict(original.run.__globals__,validate=validate,retire=retire,selection_curve=selection)
    exec(compile(body[:start]+replacement+body[stop:],'<stage1-priority-pipeline>','exec'),scope)
    return scope['run']


def run(c,p):
    with IsolatedSourceAuditEngine(cpu_workers=c['cpu_workers'],io_workers=c['io_workers']) as engine:
        selection=adopted_selection(engine)
        return adapted_run(selection)(c,p)


def detach(template,local):
    c,_=load_config(template,local);validate(c)
    root=Path(c['output']).with_name(Path(c['output']).name+'-driver')
    root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-B','-m','tools.reports.context_stage1_priority_pipeline',
          '--config',str(Path(template).resolve()),'--local',str(Path(local).resolve()),'--action','run']
    with (root/'driver.log').open('x') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=child.pid,argv=argv,started_unix=time.time(),
        process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19])
    commit_json(root/'started.json',record);return record


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['validate','detach','run'],default='validate')
    args=p.parse_args();c,provenance=load_config(args.config,args.local)
    if args.action=='detach':result=detach(args.config,args.local)
    elif args.action=='run':result=run(c,provenance)
    else:validate(c);result=dict(status='validated_only')
    print(json.dumps(result,indent=2))
