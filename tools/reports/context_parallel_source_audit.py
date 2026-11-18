"""Full Stage1 source proof using process-parallel original batch checks.

All existing transport, checkpoint, optimizer and policy validators still run.
Only batch replay is substituted with authenticated parallel results within one
call. No module monkey patch, stored PASS trust or NPU/model dispatch.
"""
import argparse
import inspect
import json
from pathlib import Path
import time
from types import FunctionType

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.parallel_evidence import workers
from tools.reports.context_parallel_batch_audit import audit_batches, identity, read_bound
from tools.reports import audit_context_infonce_training as original
from tools.reports import context_infonce_formal as formal
from tools.reports import context_infonce_source_plan as source


class BatchReplay:
    """Lazy pool uses inputs already reconstructed by the full source auditor."""
    def __init__(self, package, collection, cpu_workers):
        self.package=Path(package);self.run=Path(collection)/'run'
        self.cpu_workers=workers(cpu_workers);self.results=None;self.position=0
        self.seconds=0.;self.calls=0

    def __call__(self,batch,planned,population,task,*,policy,width,world):
        if self.results is None:
            self.config=json.loads((self.package/'config.json').read_text())
            if (self.config['experiment_kind']!='formal_infonce_source'
                    or self.config['start_update']!=0):
                raise ValueError('complete formal source required for parallel replay')
            # The caller already reconstructed this schedule from its source.
            # Every planned batch is compared again at consumption below.
            self.schedule=json.loads((self.package/'source/schedule/batches.json').read_text())
            if len(self.schedule)!=self.config['stop_update']:
                raise ValueError('complete source schedule required')
            self.population=population;self.task=task;self.width=width;self.world=world
            self.population_identity=identity(population);self.task_identity=identity(task)
            items=[(i,str(self.run/f'batch-{i:04d}.json'),file_hash(self.run/f'batch-{i:04d}.json'),p)
                   for i,p in enumerate(self.schedule)]
            self.results,self.seconds=audit_batches(items,population,task,policy=self.config['source_policy_id'],
                width=width,world=world,cpu_workers=self.cpu_workers)
        i=self.position
        if (i>=len(self.results) or planned!=self.schedule[i]
                or policy!=self.config['source_policy_id']+':instruction-update'+str(i)
                or width!=self.width or world!=self.world
                or population is not self.population or task is not self.task
                or identity(batch)!=self.results[i]['batch_semantic_sha256']
                or file_hash(self.run/f'batch-{i:04d}.json')!=self.results[i]['batch_sha256']):
            raise ValueError('parallel evidence differs from actual sequential audit inputs')
        self.position+=1;self.calls+=1
        return self.results[i]['real_queries']

    def finish(self):
        if (self.results is None or self.position!=len(self.results)
                or identity(self.population)!=self.population_identity or identity(self.task)!=self.task_identity):
            raise ValueError('incomplete replay or changed population/task')
        return dict(batches=self.calls,cpu_workers=self.cpu_workers,parallel_batch_seconds=self.seconds)


def audit(c,*,cpu_workers=4,measurements=None):
    """Return byte-for-byte JSON-equivalent full formal training proof."""
    metrics=[]
    def outputs(package,collection,*,initial_policy):
        replay=BatchReplay(package,collection,cpu_workers)
        fn=FunctionType(original.audit_outputs.__code__,dict(original.audit_outputs.__globals__,check_batch=replay))
        result=fn(package,collection,initial_policy=initial_policy)
        metrics.append(replay.finish())
        return result
    collected=FunctionType(original.audit_collected.__code__,dict(original.audit_collected.__globals__,audit_outputs=outputs),
                          argdefs=original.audit_collected.__defaults__)
    collected.__kwdefaults__=original.audit_collected.__kwdefaults__
    body=inspect.getsource(formal.audit)
    anchor='    from tools.reports.audit_context_infonce_training import audit_collected\n'
    if body.count(anchor)!=1:raise ValueError('formal audit adapter anchor changed')
    scope=dict(formal.audit.__globals__,audit_collected=collected)
    exec(compile(body.replace(anchor,''),'<parallel-full-source-audit>','exec'),scope)
    result=scope['audit'](c)
    if len(metrics)!=1:raise ValueError('exactly one full source replay required')
    if measurements is not None:measurements.extend(metrics)
    return result


def build(c,provenance,*,cpu_workers=4,measurements=None):
    fn=FunctionType(source.build.__code__,dict(source.build.__globals__,
        audit=lambda config:audit(config,cpu_workers=cpu_workers,measurements=measurements)))
    return fn(c,provenance)


def run(c,provenance):
    workers(c['cpu_workers'])
    root=Path(c['output']);root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'config.json',c);commit_json(root/'provenance.json',provenance)
    try:
        for path,digest in c['pins'].items():
            if file_hash(path)!=digest:raise ValueError('auditor source changed: '+path)
        previous=read_bound(c['source_plan']);config=read_bound(c['source_finalization'])
        measurements=[];start=time.perf_counter()
        current=build(config,previous['config_provenance'],cpu_workers=c['cpu_workers'],measurements=measurements)
        seconds=time.perf_counter()-start
        if current!=previous:raise ValueError('full source plan differs from original independent audit')
        commit_json(root/'source-plan.json',current)
        report=dict(status='parallel_full_source_audit_exact',source_plan_sha256=c['source_plan']['sha256'],
            full_proof_equal=True,updates=len(current['source_audit']['updates']),
            real_exposures=current['source_audit']['real_exposures'],full_audit_seconds=seconds,
            measurements=measurements,model_calls=0,npu_executed=False,training_dispatch_allowed=False)
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
