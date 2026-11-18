"""Ready-first four-card scheduling with independent CPU preparation/audits.

Frozen model packages, numerical auditors and metric protocols are reused.
Only this new controller owns dispatch. No retries or model-driven monitoring.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import heapq
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import FunctionType, SimpleNamespace

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths


def read(path):
    return json.loads(Path(path).read_text())


class ReadyQueue:
    """Priority among ready jobs; never preempt a running NPU job."""
    def __init__(self, events):
        self.condition = threading.Condition()
        self.pending = []
        self.busy = False
        self.failed = False
        self.serial = 0
        self.events = Path(events)
        self.events.mkdir(parents=True, exist_ok=False)
        self.event_index = 0

    def record(self, kind, **fields):
        # Caller holds the condition, including file publication ordering.
        commit_json(self.events/f'{self.event_index:06d}.json',
                    dict(event=kind, unix=time.time(), **fields))
        self.event_index += 1

    def abort(self):
        with self.condition:
            self.failed = True
            self.record('dispatch_stopped')
            self.condition.notify_all()

    def check(self):
        with self.condition:
            if self.failed:
                raise RuntimeError('pipeline failure: no further dispatch')

    @contextmanager
    def slot(self, priority, tag):
        with self.condition:
            token = (priority, self.serial, tag)
            self.serial += 1
            heapq.heappush(self.pending, token)
            self.record('npu_ready', tag=tag, priority=priority)
            while not self.failed and (self.busy or self.pending[0] != token):
                self.condition.wait()
            if self.failed:
                self.pending.remove(token)
                heapq.heapify(self.pending)
                raise RuntimeError('pipeline failure: queued dispatch cancelled')
            heapq.heappop(self.pending)
            self.busy = True
            self.record('npu_acquired', tag=tag)
        try:
            yield
        finally:
            with self.condition:
                self.busy = False
                self.record('npu_released', tag=tag)
                self.condition.notify_all()


def launcher(module, queue, priority, *,producer, authorization):
    """Scoped adaptation of the shared lifecycle; no module monkey patch.

    Both original admission checks run before requesting NPU ownership.
    The original flock, idleness/image checks and remote package hashes remain.
    Ownership ends only after exact docker wait AND inspect, before collection.
    """
    from tools.reports import context_replica_training_operations as shared
    from tools.reports.context_cached_storage_operations import command, storage_preflight
    source = inspect.getsource(shared.launch)
    begin = "        check = storage + '\\ncheck_paths('"
    end = "        checked(['python3', '-c', check])"
    if source.count(begin) != 1 or source.count(end) != 1:
        raise ValueError('shared storage anchors changed')
    left, right = source.index(begin), source.index(end, source.index(begin))
    source = source[:left]+"        check = storage + '\\n' + storage_preflight(config,root)\n"+source[right:]
    before = "    with (base / (c['tag'] + '.lock')).open('a') as owner, Path(c['lease']).open('r') as lease:"
    after = "    validate(c)\n    with queue.slot(priority, c['tag']), (base / (c['tag'] + '.lock')).open('a') as owner, Path(c['lease']).open('r') as lease:"
    if source.count(before) != 1 or source.count('        validate(c)\n') != 1:
        raise ValueError('shared admission/ownership anchors changed')
    source = source.replace('        validate(c)\n', '').replace(before, after)
    before = "        write(base / (c['tag'] + '-inspect.json'), inspected)\n"
    if source.count(before) != 1:
        raise ValueError('shared exact inspect anchor changed')
    source = source.replace(before, before + "        if code != 0: queue.abort()\n")
    # inspect and wait still occur inside the exclusive lease. CPU collection
    # and runtime proof checks are dedented to run after both locks are released.
    start = source.index('        result = collect(record,')
    source = source[:start]+''.join(line[4:] if line.startswith('    ') else line
                                   for line in source[start:].splitlines(keepends=True))
    scope = dict(shared.launch.__globals__, validate=module.validate,
        command=lambda c,m,cfg: command(c,m,cfg,producer=producer),
        storage_preflight=lambda cfg,root: storage_preflight(cfg,root,producer=producer),
        queue=queue, priority=priority)
    exec(compile(source, '<async-shared-lifecycle-v1>', 'exec'), scope)

    def launch(c, provenance):
        if c.get(authorization) is not True:
            raise ValueError('explicit dispatch authorization required: '+authorization)
        queue.check()
        try:
            result = scope['launch'](dict(c,finite_training_dispatch_allowed=True), provenance)
            if result['exit_code'] != 0:
                raise ValueError('NPU execution failed; collection preserved')
            status = ('transfer_collected_independent_audit_required' if authorization=='transfer_dispatch_allowed'
                      else 'selection_collected_independent_audit_required' if producer
                      else 'collected_independent_cached_training_audit_required')
            return dict(result,status=status)
        except BaseException:
            queue.abort()
            raise
    return launch


def operation(module, ops, provenance, queue, priority, *,producer=False, authorization):
    launch = launcher(module, queue, priority, producer=producer, authorization=authorization)
    return FunctionType(module.run.__code__, dict(module.run.__globals__,launch=launch))(ops,provenance)


def locations(c, name):
    return dict(package=str(Path(c['prepared_root'])/name),
        operations_root=str(Path(c['operations_root'])/name),
        remote_root=str(Path(c['remote_root'])/name),tag=c['tag_prefix']+'-'+name)


def checked_source(binding):
    if file_hash(binding['path']) != binding['sha256']:
        raise ValueError('controller input changed: '+binding['path'])
    return read(binding['path'])


def validate(c):
    if c.get('dispatch_allowed') is not True or c.get('monitor_interval_seconds') != 900:
        raise ValueError('explicit dispatch and 900-second observation required')
    if c.get('cpu_workers') != 4 or c.get('evaluation_workers') != 2:
        raise ValueError('bounded CPU4 / evaluation2 controller required')
    for name,digest in c['pins'].items():
        if file_hash(name) != digest:
            raise ValueError('frozen controller dependency changed: '+name)
    old = {name:checked_source(value) for name,value in c['sources'].items()}
    from tools.reports import context_replica_transfer_sequence as transfer
    transfer.validate(old['transfer'])  # Complete prior identity/path chain.
    if c['deployment'] != old['selection']['deployment']:
        raise ValueError('same original four-card deployment required')
    expected_endpoint={k:old['selection']['points'][-1][k] for k in ('package','remote_root','manifest_sha256')}
    if c.get('preprepared_stage1_endpoint') != expected_endpoint:
        raise ValueError('only exact preprepared Stage1 endpoint reuse is allowed')
    if set(old) != {'finite','efficiency','selection','formal','grpo','transfer'}:
        raise ValueError('complete original queue required')
    expected_drivers = [str(Path(old[name]['output']).with_name(Path(old[name]['output']).name+'-driver')/'started.json')
                        for name in ('transfer','grpo','formal','selection','efficiency','finite')]
    if [b['path'] for b in c['retire_drivers']] != expected_drivers:
        raise ValueError('exact downstream-first queue retirement required')
    for binding in c['retire_drivers']:
        checked_source(binding)
    if old['selection']['point_template']['parent'] != read(old['finite']['reference_predecessor']['config'])['reference_job']['parent']:
        raise ValueError('same Stage1 source/runtime required')
    roots=[Path(c[k]).resolve() for k in ('output','prepared_root','operations_root','remote_root')]
    for i,path in enumerate(roots):
        if any(path.is_relative_to(p) or p.is_relative_to(path) for p in roots[:i]):
            raise ValueError('disjoint output roots required')
    protected=[]
    def paths(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k in ('output','package','operations_root','remote_root','prepared_root','scores_root') and isinstance(v,str):
                    protected.append(Path(v).resolve())
                elif isinstance(v,(dict,list)):paths(v)
        elif isinstance(value,list):
            for v in value:paths(v)
    for value in old.values():paths(value)
    for path in roots:
        if any(path.is_relative_to(p) or p.is_relative_to(path) for p in protected):
            raise ValueError('new controller cannot overwrite previous artifacts')
    check_paths({k:c[k] for k in ('output','prepared_root','operations_root')})
    return old


def exact_process(record):
    proc=Path('/proc')/str(record['pid'])
    stat=(proc/'stat').read_text().split(') ',1)[1].split()
    argv=(proc/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
    if stat[19]!=record['process_start_ticks'] or argv!=record['argv']:
        raise ValueError('driver process identity changed')
    return stat[0]


def retire(c, old):
    """Explicit user-requested queue replacement, only with all NPUs idle.

    Stop downstream waiters first. Never rewrite their terminal records or
    signal remote model processes. SIGSTOP closes the dispatch race while
    checking lease, child processes, local attempts and actual NPU idleness.
    """
    import fcntl
    import select
    records=[checked_source(b) for b in c['retire_drivers']]
    handles=[]
    try:
        for record in records:
            fd=os.pidfd_open(record['pid']);handles.append((fd,record))
            exact_process(record)
            signal.pidfd_send_signal(fd,signal.SIGSTOP)
        for _,record in handles:
            deadline=time.monotonic()+2
            while exact_process(record) not in ('T','t'):
                if time.monotonic() >= deadline:
                    raise ValueError('driver did not stop for safe queue replacement')
                time.sleep(.01)
            child=Path(f"/proc/{record['pid']}/task/{record['pid']}/children").read_text().strip()
            if child:raise ValueError('driver owns child work; retain old queue')
        dep=old['selection']['deployment']
        with Path(dep['lease']).open('r') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result=subprocess.run(['ssh','-o','ConnectTimeout=10',dep['host'],'npu-smi info'],
                capture_output=True,text=True,check=True,timeout=55)
            if any(f'No running processes found in NPU {n}' not in result.stdout for n in (4,5,6,7)):
                raise ValueError('NPU work active; old queue retained')
            for name in ('tail','recovery'):
                stage=old['finite']['stages'][name];root=Path(stage['operations_root'])
                if (root/(stage['tag']+'-attempt.json')).exists() or (root/(stage['tag']+'-launch.json')).exists():
                    raise ValueError('finite NPU attempt already exists; no automatic migration')
            for name in ('efficiency','formal'):
                stage=old[name]['experiment'];root=Path(stage['operations_root'])
                if (root/(stage['tag']+'-attempt.json')).exists():raise ValueError('downstream attempt exists')
            for name in ('selection','grpo'):
                for item in old[name]['points']:
                    if (Path(item['operations_root'])/(item['tag']+'-attempt.json')).exists():
                        raise ValueError('evaluation attempt already exists')
            commit_json(Path(c['output'])/'retirement-intent.json',dict(records=records,unix=time.time(),
                reason='user_requested_async_cpu_npu_scheduling',npu_idle_devices=[4,5,6,7]))
            for fd,_ in handles:
                signal.pidfd_send_signal(fd,signal.SIGTERM)
                signal.pidfd_send_signal(fd,signal.SIGCONT)
            for fd,_ in handles:
                poll=select.poll();poll.register(fd,select.POLLIN)
                if not poll.poll(10000):raise RuntimeError('old controller failed to exit')
            commit_json(Path(c['output'])/'retirement-complete.json',dict(retired_pids=[r['pid'] for _,r in handles],
                old_artifacts_preserved=True,remote_processes_signalled=False,unix=time.time()))
    except BaseException:
        for fd,_ in handles:
            try:signal.pidfd_send_signal(fd,signal.SIGCONT)
            except ProcessLookupError:pass
        raise
    finally:
        for fd,_ in handles:os.close(fd)


def selection_curve(c, template, steps, provenance, queue, *,grpo=False):
    if grpo:
        from tools.reports import prepare_context_cached_grpo_selection as preparation
        from tools.reports import context_cached_grpo_selection_operations as point
        from tools.reports import collect_context_cached_grpo_curve as curve
    else:
        from tools.reports import prepare_context_replica_selection as preparation
        from tools.reports import context_replica_selection_operations as point
        from tools.reports import collect_context_replica_infonce_curve as curve
    prefix='grpo' if grpo else 'infonce';root=Path(c['output'])/prefix
    root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'template.json',template)
    def one(step):
        queue.check();where=locations(c,f'{prefix}-{step:04d}')
        spec=dict(template,source_step=step,output=where['package'],remote_root=where['remote_root'])
        existing=c.get('preprepared_stage1_endpoint') if not grpo and step==steps[-1] else None
        if existing:
            # Immutable, never previously launched endpoint package. New
            # operations records remain isolated; audit all bytes on reuse.
            where.update(package=existing['package'],remote_root=existing['remote_root'])
            spec.update(output=where['package'],remote_root=where['remote_root'],manifest_sha256=existing['manifest_sha256'])
            preparation.audit(spec,provenance)
        else:
            spec['manifest_sha256']=preparation.prepare(spec,provenance)['manifest_sha256']
        ops=dict(c['deployment'],selection_package=spec,manifest_sha256=spec['manifest_sha256'],
            **{k:where[k] for k in ('package','operations_root','tag')},selection_dispatch_allowed=True)
        proof=operation(point,ops,provenance,queue,0,producer=True,authorization='selection_dispatch_allowed')
        expected='complete_replica_cached_grpo_selection_audited' if grpo else 'complete_replica_infonce_selection_audited'
        if proof['status']!=expected or proof['source_step']!=step or proof['runtime_provenance_verified'] is not True:
            raise ValueError('full independent selection point audit required')
        path=Path(where['operations_root'])/'finalization/audit-config.json'
        entry=dict(step=step,audit_config=str(path),audit_config_sha256=file_hash(path))
        commit_json(root/f'completed-{step:04d}.json',entry)
        return entry
    entries=[]
    # Endpoint first delivers the most useful baseline early; the published
    # curve remains in the preregistered order and uses every registered point.
    order=[steps[-1],*steps[:-1]]
    with ThreadPoolExecutor(max_workers=c['evaluation_workers'],thread_name_prefix=prefix) as pool:
        futures={pool.submit(one,s):s for s in order}
        try:
            for future in as_completed(futures):entries.append(future.result())
        except BaseException:
            queue.abort()
            for future in futures:future.cancel()
            raise
    entries.sort(key=lambda e:steps.index(e['step']))
    cc=dict(template,evaluations=entries,output=str(root/'curve'))
    commit_json(root/'curve-config.json',cc);result=curve.collect(cc,provenance)
    commit_json(root/'terminal.json',dict(exit_code=0,status=result['status'],points=len(entries)))
    return cc,result


def finite_and_efficiency(c,old,provenance,queue):
    from tools.reports import context_cached_acceptance_sequence as finite
    from tools.reports import context_cached_training_operations as training
    from tools.reports import context_cached_recovery_operations as recovery
    from tools.reports import context_cached_efficiency as efficiency
    from tools.reports.context_cached_efficiency_successor import cost_config
    from tools.reports.context_cached_efficiency_cost import run as report_cost
    from tools.reports.context_release_deployment import inherited
    prior=old['finite'];ref=read(prior['reference_predecessor']['audit_config'])
    refops=read(prior['reference_predecessor']['config'])
    prefix_root=Path(prior['stages']['prefix']['operations_root'])/'finalization'
    prefix=read(prefix_root/'audit-config.json')
    # Reconstruct evidence independently; completed prefix is not rerun on NPU.
    proof=training.audit(prefix,provenance)
    if proof['status']!=finite.STATUSES['prefix']:raise ValueError('prefix admission failed')
    evidence={'prefix':prefix}
    for name,module in [('tail',training),('recovery',recovery)]:
        proxy=SimpleNamespace(launch=launcher(module,queue,10,producer=False,authorization='finite_training_dispatch_allowed'),
                              audit=module.audit)
        fn=FunctionType(finite.run_stage.__code__,dict(finite.run_stage.__globals__,
            training=proxy if name=='tail' else training,recovery=proxy if name=='recovery' else recovery))
        bound,proof=fn(name,locations(c,name),refops,ref,prefix,provenance)
        evidence[name]=bound
    stage=locations(c,'efficiency');root=Path(c['output'])/'efficiency';root.mkdir(parents=True,exist_ok=False)
    spec=dict(parent=prefix['training_package']['parent'],reference_job=prefix['training_package']['reference_job'],
        acceptance=evidence,acceptance_kind='efficiency',output=stage['package'],remote_root=stage['remote_root'])
    commit_json(root/'preparation-config.json',spec)
    spec['manifest_sha256']=efficiency.package_action(spec,provenance)['manifest_sha256']
    ops=dict(inherited(refops),training_package=spec,reference_audit=ref,manifest_sha256=spec['manifest_sha256'],
        **{k:stage[k] for k in ('package','operations_root','tag')},efficiency_dispatch_allowed=True)
    commit_json(root/'deployment.json',ops)
    launcher(efficiency,queue,20,producer=False,authorization='efficiency_dispatch_allowed')(ops,provenance)
    bound=finite.audit_config(ops,ref);commit_json(root/'audit-config.json',bound)
    proof=efficiency.audit(bound,provenance)
    if proof['status']!='cached_100step_efficiency_outputs_audited':raise ValueError('full100 efficiency audit required')
    commit_json(root/'audit.json',proof)
    measure=cost_config(bound,root/'cost.json');commit_json(root/'cost-config.json',measure);report_cost(measure,provenance)
    commit_json(root/'terminal.json',dict(exit_code=0,status=proof['status'],updates=100))
    return bound


def formal_training(c,bound,source_curve,provenance,queue):
    from tools.reports import context_cached_formal as formal
    from tools.reports.context_release_deployment import inherited
    from tools.reports.context_cached_formal_successor import monitored_training
    prior=bound['training_package'];where=locations(c,'formal');root=Path(c['output'])/'formal'
    root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'source-admission.json',dict(curve=source_curve,stage1_endpoint_initialization=True))
    spec=dict(parent=prior['parent'],reference_job=prior['reference_job'],efficiency_audit=bound,
        acceptance_kind='formal_source',evaluation_steps=list(formal.STEPS),
        output=where['package'],remote_root=where['remote_root'])
    commit_json(root/'preparation-config.json',spec)
    spec['manifest_sha256']=formal.package_action(spec,provenance)['manifest_sha256']
    ops=dict(inherited(bound['deployment']),training_package=spec,reference_audit=bound['reference_audit'],
        manifest_sha256=spec['manifest_sha256'],**{k:where[k] for k in ('package','operations_root','tag')},
        formal_training_dispatch_allowed=True)
    commit_json(root/'deployment.json',ops)
    proxy=SimpleNamespace(run=lambda o,p:operation(formal,o,p,queue,20,authorization='formal_training_dispatch_allowed'))
    fn=FunctionType(monitored_training.__code__,dict(monitored_training.__globals__,formal=proxy))
    result=fn(ops,provenance,root)
    path=Path(ops['operations_root'])/'finalization/audit-config.json'
    finalization=dict(training_audit_config=str(path),training_audit_config_sha256=file_hash(path))
    return finalization,ops,result


def transfers(c,old,source_cc,source_result,provenance,queue):
    from tools.reports import context_replica_transfer_sequence as sequence
    from tools.reports.context_cached_grpo_transfer_source import admit
    source_eval=source_result['source_evaluation'];bindings={};common=None
    for role in sequence.ROLES:
        request=dict(source_evaluation=source_eval,source_curve_config=source_cc,
            policy_role=role,policy_id=source_result['roles'][role]['policy_id'])
        proof=admit(request)
        if proof['binding']!=source_result['roles'][role]:raise ValueError('source role binding differs')
        current=proof['source_operations']
        if common is not None and current!=common:raise ValueError('different source training')
        common=current;bindings[role]=request
    parent=common['training_package']['parent'];matrices={}
    def one(dataset,slug,role,index):
        binding=bindings[role];item=old['transfer']['datasets'][dataset];where=locations(c,f'{slug}-role-{index}')
        spec=dict(parent=parent,source_admission=binding,dataset=dataset,
            portable_full_inputs=item['portable_full_inputs'],portable_parity_inputs=item['portable_parity_inputs'],
            output=where['package'],remote_root=where['remote_root'])
        if dataset=='SLURP_clean_TEST60':spec['source_release']=item['original_evaluation']['source_release']
        spec['manifest_sha256']=sequence.preparation.prepare(spec,provenance)['manifest_sha256']
        ops=dict(c['deployment'],transfer_package=spec,manifest_sha256=spec['manifest_sha256'],
            **{k:where[k] for k in ('package','operations_root','tag')},transfer_dispatch_allowed=True)
        result=operation(sequence.operations,ops,provenance,queue,0,producer=True,authorization='transfer_dispatch_allowed')
        if result['status']!='complete_replica_transfer_audited' or result['target_scored'] is not False:
            raise ValueError('complete unscored transfer inference required')
        return read(Path(ops['operations_root'])/'finalization/audit-config.json')
    # Submit every unique frozen policy once; target gold is scored only after
    # all inference is complete and independently audited.
    with ThreadPoolExecutor(max_workers=c['evaluation_workers']) as pool:
        futures={}
        for dataset,slug in zip(sequence.DATASETS,sequence.SLUGS,strict=True):
            for index,role in enumerate(sequence.ROLES):
                key=(dataset,bindings[role]['policy_id'])
                if key not in futures:futures[key]=pool.submit(one,dataset,slug,role,index)
        try:executions={key:f.result() for key,f in futures.items()}
        except BaseException:
            queue.abort()
            for f in futures.values():f.cancel()
            raise
    for dataset,slug in zip(sequence.DATASETS,sequence.SLUGS,strict=True):
        item=old['transfer']['datasets'][dataset]
        entries={role:dict(policy_id=bindings[role]['policy_id'],operations=executions[(dataset,bindings[role]['policy_id'])])
                 for role in sequence.ROLES}
        spec=dict(entries=entries,original_evaluation=item['original_evaluation'],scoring=item['scoring'],
                  output=str(Path(c['output'])/'transfer-scores'/slug))
        matrices[dataset]=sequence.scoring.collect(spec,provenance)
    return matrices


def run(c,provenance):
    old=validate(c);root=Path(c['output']);root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'config.json',c);commit_json(root/'provenance.json',provenance)
    queue=ReadyQueue(root/'events')
    try:
        retire(c,old)
        template=old['selection']['point_template']
        from tools.reports.context_infonce_selection import STEPS
        with ThreadPoolExecutor(max_workers=2,thread_name_prefix='independent-branches') as branches:
            selection=branches.submit(selection_curve,c,template,STEPS,provenance,queue)
            efficiency=branches.submit(finite_and_efficiency,c,old,provenance,queue)
            results={}
            try:
                for future in as_completed([selection,efficiency]):results[future]=future.result()
            except BaseException:
                queue.abort()
                raise
        source_cc,source_result=results[selection]
        finalization,ops,_=formal_training(c,results[efficiency],source_cc,provenance,queue)
        from tools.reports import context_cached_grpo_source_plan as source
        from tools.reports.context_cached_formal import STEPS as GRPO_STEPS
        plan=source.build(finalization,provenance);plan_path=root/'grpo-source-plan.json';commit_json(plan_path,plan)
        template=dict(parent=ops['training_package']['parent'],source_plan=str(plan_path),source_plan_sha256=file_hash(plan_path),
            source_finalization=finalization,selection_inputs=template['selection_inputs'])
        cc,result=selection_curve(c,template,GRPO_STEPS,provenance,queue,grpo=True)
        result=transfers(c,old,cc,result,provenance,queue)
        commit_json(root/'results.json',result)
        commit_json(root/'terminal.json',dict(exit_code=0,status='async_simple_pipeline_completed',restart=False))
    except BaseException as exc:
        queue.abort()
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False,preserve_handles=True))
        raise


def detach(template,local):
    c,_=load_config(template,local);validate(c)
    root=Path(c['output']).with_name(Path(c['output']).name+'-driver')
    check_paths({'driver':root});root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-B','-m',__name__ if __name__!='__main__' else 'tools.reports.context_async_pipeline',
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
    p.add_argument('--action',choices=['validate','run','detach'],default='validate');a=p.parse_args()
    c,provenance=load_config(a.config,a.local)
    if a.action=='detach':result=detach(a.config,a.local)
    elif a.action=='run':result=run(c,provenance)
    else:validate(c);result=dict(status='validated_only',npu_executed=False)
    print(json.dumps(result,indent=2))
