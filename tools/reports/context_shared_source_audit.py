"""Shared content-bound batch proofs in full source/selection preparation.

Scoped function adapters leave frozen modules and NPU packages untouched. Full
source audits still verify transport, all checkpoints and all optimizer states.
"""
import argparse
import inspect
import json
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import time
from types import FunctionType,SimpleNamespace

from multimodal_jev.audit_singleflight import AuditSingleFlight
from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.parallel_evidence import inventory, workers
from tools.reports.context_parallel_batch_audit import identity,read_bound,audit_batches
from tools.reports import context_parallel_source_audit as parallel


class SharedBatchReplay(parallel.BatchReplay):
    def __init__(self,*args,store,io_workers,events,event_lock,**kwargs):
        super().__init__(*args,**kwargs)
        self.store=store;self.io_workers=io_workers;self.events=events;self.event_lock=event_lock

    def __call__(self,batch,planned,population,task,*,policy,width,world):
        if self.results is None:
            self.config=json.loads((self.package/'config.json').read_text())
            if self.config['experiment_kind']!='formal_infonce_source' or self.config['start_update']!=0:
                raise ValueError('complete formal source required')
            self.schedule=json.loads((self.package/'source/schedule/batches.json').read_text())
            if len(self.schedule)!=self.config['stop_update']:raise ValueError('complete source schedule required')
            self.population=population;self.task=task;self.width=width;self.world=world
            self.population_identity=identity(population);self.task_identity=identity(task)
            # Fresh byte hashes, exact ordered schedule, complete population/task,
            # and all project checker dependencies participate in the key.
            start=time.perf_counter()
            project=Path(__file__).resolve().parents[2]
            # Read-only code inventory is intentionally conservative. Includes
            # local helpers, not just check_batch's immediate source module.
            code_paths=sorted({str(p) for folder in (project/'src/multimodal_jev',project/'tools/reports')
                               for p in folder.glob('*.py')})
            batch_paths=[str(self.run/f'batch-{i:04d}.json') for i in range(len(self.schedule))]
            paths=code_paths+batch_paths
            with ThreadPoolExecutor(max_workers=self.io_workers) as pool:
                digests=dict(zip(paths,pool.map(file_hash,paths),strict=True))
            code={p:digests[p] for p in code_paths}
            items=[(i,path,digests[path],p) for i,(path,p) in enumerate(zip(batch_paths,self.schedule,strict=True))]
            key=identity(dict(items=items,population=population,task=task,
                policy=self.config['source_policy_id'],width=width,world=world,code=code))
            def compute():
                return audit_batches(items,population,task,policy=self.config['source_policy_id'],
                    width=width,world=world,cpu_workers=self.cpu_workers)
            (self.results,self.seconds),hit=self.store.get(key,compute)
            with self.event_lock:self.events.append(dict(cache_hit=hit,content_key=key,
                elapsed_seconds=time.perf_counter()-start,batches=len(items)))
        return super().__call__(batch,planned,population,task,policy=policy,width=width,world=world)


class SourceAuditEngine:
    """One explicit instance per controller; shared bounded CPU budget."""
    def __init__(self,*,cpu_workers=4,io_workers=4,max_entries=2):
        self.cpu_workers=workers(cpu_workers);self.io_workers=workers(io_workers)
        self.store=AuditSingleFlight(max_entries=max_entries,max_running=1)
        self.events=[];self.lock=threading.Lock()

    def audit(self,c,*,measurements=None):
        def replay(*args):
            return SharedBatchReplay(*args,store=self.store,io_workers=self.io_workers,
                events=self.events,event_lock=self.lock)
        fn=FunctionType(parallel.audit.__code__,dict(parallel.audit.__globals__,BatchReplay=replay))
        return fn(c,cpu_workers=self.cpu_workers,measurements=measurements)

    def build(self,c,provenance):
        from tools.reports import context_infonce_source_plan as source
        return FunctionType(source.build.__code__,dict(source.build.__globals__,audit=self.audit))(c,provenance)

    def reconstruct(self,c):
        from tools.reports import context_infonce_selection as selection
        return FunctionType(selection.reconstruct_plan.__code__,dict(selection.reconstruct_plan.__globals__,build=self.build))(c)

    def preparation(self):
        """Connect the engine through selection -> runtime -> source -> plan.

        Does not modify module globals; caller explicitly supplies this object.
        Original reconstruction, manifest/hash and numerical checks remain.
        """
        from tools.reports import context_infonce_stage2_source as source
        from tools.reports import prepare_context_stage2_runtime as runtime
        from tools.reports import prepare_context_replica_selection as selection
        source_audit=partial(source.audit,reconstruct=self.reconstruct)
        runtime_audit=partial(runtime.audit,source_auditor=source_audit)
        inputs=FunctionType(selection.inputs.__code__,dict(selection.inputs.__globals__,reconstruct_plan=self.reconstruct,audit_parent=runtime_audit))
        prepare=FunctionType(selection.prepare.__code__,dict(selection.prepare.__globals__,inputs=inputs))
        audit=FunctionType(selection.audit.__code__,dict(selection.audit.__globals__,prepare=prepare))
        return SimpleNamespace(prepare=prepare,audit=audit)


def run(c,provenance):
    root=Path(c['output']);root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'config.json',c);commit_json(root/'provenance.json',provenance)
    try:
        for name,digest in c['pins'].items():
            if file_hash(name)!=digest:raise ValueError('engine code changed: '+name)
        old=read_bound(c['source_plan']);finalization=read_bound(c['source_finalization'])
        engine=SourceAuditEngine(cpu_workers=c['cpu_workers'],io_workers=c['io_workers'])
        times=[]
        for i in range(2):
            start=time.perf_counter();result=engine.build(finalization,old['config_provenance']);times.append(time.perf_counter()-start)
            if result!=old:raise ValueError('full source proof differs from original')
            commit_json(root/f'source-plan-{i}.json',result)
        if engine.store.executions!=1 or engine.store.hits!=1:
            raise ValueError('same full source must share one batch computation')
        # Exercise actual selection package reconstruction through every wrapper.
        package=read_bound(c['selection_spec'])
        start=time.perf_counter();proof=engine.preparation().audit(package,provenance);selection_seconds=time.perf_counter()-start
        if proof['status']!='replica_infonce_selection_package_audited_not_executed':raise ValueError('selection reconstruction failed')
        commit_json(root/'selection-audit.json',proof)
        report=dict(status='shared_source_and_selection_chain_verified',full_proof_equal=True,
            source_audit_seconds=times,selection_audit_seconds=selection_seconds,batch_computations=engine.store.executions,
            batch_reuses=engine.store.hits,events=engine.events,model_calls=0,npu_executed=False,
            training_dispatch_allowed=False)
        commit_json(root/'report.json',report);commit_json(root/'terminal.json',dict(exit_code=0,status=report['status']))
        return report
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False));raise


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    print(json.dumps(run(c,provenance),indent=2))
